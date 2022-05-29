import numpy as np
from tqdm import tqdm
import cv2
from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from bisect import bisect


def regularize_contour(contour, definition=1000):
    contour = np.array(contour)
    first_point = np.argmin(np.linalg.norm(contour[:-1], axis=1))
    if first_point != 0:
        contour = np.concatenate((contour[:-1][first_point:], contour[:-1][:first_point]))
        contour = np.concatenate((contour, [contour[0]]))

    lengths = np.cumsum(np.linalg.norm(np.array([np.diff(contour[:, 0]), np.diff(contour[:, 1])]).T, axis=1))
    perimeter = lengths[-1]
    points = []
    steps = np.arange(0, perimeter+perimeter/(2*definition), perimeter/definition)
    indices = list(map(lambda x: bisect(lengths, x), steps))
    for i, index in enumerate(indices):
        step = steps[i]
        if index==0:
            t = step / lengths[index]
        elif index == len(contour)-1:
            t = (step - lengths[index-1]) / (lengths[-1] + lengths[0] - lengths[index-1])
        else:
            t = (step - lengths[index-1]) / (lengths[index] - lengths[index-1])
        if index == len(contour)-1:
            points.append(contour[1] * t + contour[index] * (1-t))
        else:
            points.append(contour[index+1] * t + contour[index] * (1-t))

    points = np.array(points)
    first_point = np.argmin(np.linalg.norm(points, axis=1))
    if first_point != 0:
        points = np.concatenate((points[first_point:], points[:first_point]))
    return points

def fit_interpolator(target_contour, current_contour, target_width, target_height, bounds):
    target_contour_ = target_contour.copy()
    current_contour_ = current_contour.copy()
    target_contour_[:, 0] /= target_width
    target_contour_[:, 1] /= target_height
    current_contour_[:, 0] -= bounds[0]
    current_contour_[:, 1] -= bounds[1]
    current_contour_[:, 0] /= bounds[2]
    current_contour_[:, 1] /= bounds[3]
    return RBFInterpolator(target_contour_,  current_contour_, kernel="cubic", smoothing=0.01)

def find_image_transformation(interpolator, grid, target_width, target_height, bounds):
    reshaped_grid = grid.reshape(-1, 2)
    reshaped_grid[:, 0] /= target_width
    reshaped_grid[:, 1] /= target_height
    points = interpolator(reshaped_grid)
    reshaped_grid[:, 0] *= target_width
    reshaped_grid[:, 1] *= target_height
    points[:, 0] *= bounds[2]
    points[:, 1] *= bounds[3]
    points[:, 0] += bounds[0]
    points[:, 1] += bounds[1]
    return points, reshaped_grid

def warp_image(grid, points, image, reshaped_grid):
    warped_image = np.zeros(tuple(np.array(grid.shape)[:-1])[::-1]+(3,)).astype(np.uint8)
    x_decimal = points[:, 0] - np.floor(points[:, 0])
    y_decimal = points[:, 1] - np.floor(points[:, 1])
    x_ceil = np.ceil(points[:, 0]).astype(int)
    y_ceil = np.ceil(points[:, 1]).astype(int)
    x_floor = np.floor(points[:, 0]).astype(int)
    y_floor = np.floor(points[:, 1]).astype(int)
    y_floor[y_floor>=image.shape[0]] = image.shape[0]-1
    x_floor[x_floor>=image.shape[1]] = image.shape[1]-1
    y_floor[y_floor<0] = 0
    x_floor[x_floor<0] = 0
    y_ceil[y_ceil>=image.shape[0]] = image.shape[0]-1
    x_ceil[x_ceil>=image.shape[1]] = image.shape[1]-1
    y_ceil[y_ceil<0] = 0
    x_ceil[x_ceil<0] = 0
    distances = np.array([
        1/np.linalg.norm(np.array([x_decimal, y_decimal]).T, axis=1),
        1/np.linalg.norm(np.array([1-x_decimal, y_decimal]).T, axis=1),
        1/np.linalg.norm(np.array([x_decimal, 1-y_decimal]).T, axis=1),
        1/np.linalg.norm(np.array([1-x_decimal, 1-y_decimal]).T, axis=1),
    ]).T
    for i, v in enumerate(np.sum(distances, 1)):
        distances[i] /= v

    raveled_warped_image = (
        image[y_floor, x_floor] * distances[:, 0, None]
        + image[y_floor, x_ceil] * distances[:, 1, None]
        + image[y_ceil, x_floor] * distances[:, 2, None]
        + image[y_ceil, x_ceil] * distances[:, 3, None]
    )
    nans = np.isnan(distances).any(1)
    grid_coords = np.round(reshaped_grid, 1).astype(int)
    warped_image[grid_coords[:, 1], grid_coords[:, 0]] = raveled_warped_image.astype(np.uint8)
    points[:, 1][points[:, 1]>=image.shape[0]] = image.shape[0]-1
    points[:, 0][points[:, 0]>=image.shape[1]] = image.shape[1]-1
    points[:, 1][points[:, 1]<0] = 0
    points[:, 0][points[:, 0]<0] = 0
    warped_image[grid_coords[nans, 1], grid_coords[nans, 0]] = image[points.astype(int)[:, 1], points.astype(int)[:, 0]][nans]
    return warped_image

def warp_image_into_contour(image, target_contour, current_contour):
    target_width, target_height = np.max(target_contour, 0)
    bounds = [min(current_contour[:, 0]), min(current_contour[:, 1])]
    bounds = bounds + [max(current_contour[:, 0]) - bounds[0], max(current_contour[:, 1]) - bounds[1]]
    grid = np.array(np.meshgrid(np.arange(0, target_width, 1), np.arange(0, target_height, 1))).T
    interpolator = fit_interpolator(target_contour, current_contour, target_width, target_height, bounds)
    points, reshaped_grid = find_image_transformation(interpolator, grid, target_width, target_height, bounds)
    return warp_image(grid, points, image, reshaped_grid)

def normalize_target_contour(target_contour):
    offset_width, offset_height = np.min(target_contour, 0)
    target_contour[:, 0] -= offset_width
    target_contour[:, 1] -= offset_height
    return target_contour

def morph_image_into_shape(image, polygon, target_polygon):
    polygon = polygon.simplify(polygon.length*0.00001)

    contour = np.array(polygon.exterior.coords)
    current_contour = regularize_contour(contour)

    contour = np.array(target_polygon.exterior.coords)
    target_contour = regularize_contour(contour)
    target_contour = normalize_target_contour(target_contour)

    warped_image = warp_image_into_contour(image, target_contour, current_contour)
    return warped_image.astype(np.uint8)

import cv2
from matplotlib import pyplot as plt
def get_image_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = (255*(gray>2)).astype(np.uint8)
    contours, hierarchies = cv2.findContours(
        thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_areas = []
    for contour in contours:
        try:
            contours_areas.append(Polygon(np.squeeze(contour)).area)
        except:
            contours_areas.append(0)
    image_contour = np.squeeze(contours[np.argmax(contours_areas)])
    # plt.scatter(current_shape[:, 0], current_shape[:, 1])
    return image_contour

if __name__ =="__main__":
    image = cv2.imread("input_image.png")
    plt.imshow(image)
    current_shape = get_image_contour(image)
    target_image = cv2.imread("target_image.png")
    plt.imshow(target_image)
    target_shape = get_image_contour(target_image)
    warped_image = morph_image_into_shape(image, Polygon(current_shape), Polygon(target_shape))
    plt.imshow(warped_image)
