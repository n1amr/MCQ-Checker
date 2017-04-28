import math
import re
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(img, msg=None, unstack=False, complete=False):
    n = img.shape[0]
    n3 = n // 3
    x = 5

    if unstack:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if msg is not None:
            plt.title(msg)

        img = np.hstack([img[:n3],
                         np.ones((n3, x, 3), dtype='uint8'),
                         img[n3:2 * n3],
                         np.ones((n3, x, 3), dtype='uint8'),
                         img[2 * n3:]])
        plt.imshow(img)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()
    else:
        for i in [2]:
            if msg is not None:
                plt.title(msg)

            if complete:
                segment = img
            else:
                segment = img[i * n // x: (i + 2) * n // x, :]

            plt.imshow(segment, 'gray')
            plt.get_current_fig_manager().full_screen_toggle()
            plt.show()


def inspect_img(img):
    print(f'Type: {type(img)}')
    print(f'Shape: {img.shape}')
    print(f'dtype: {img.dtype}')
    print(f'Range: {img.min(), img.max()}')


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


kp1, des1 = None, None


def deskew_image(img_original, img_skewed, debug=False):
    surf = cv2.xfeatures2d.SURF_create(400)

    global kp1, des1
    if kp1 is None:
        kp1, des1 = surf.detectAndCompute(img_original, None)
    kp2, des2 = surf.detectAndCompute(img_skewed, None)

    # FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    # search_params = dict(checks=50)
    search_params = dict(checks=10)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt
                              for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt
                              for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi

        if debug:
            print(
                f'Calculated scale difference: {scaleRecovered:0.2f}\n'
                f'Calculated rotation difference: {thetaRecovered:0.2f}')

        img_deskewed = cv2.warpPerspective(img_skewed,
                                           np.linalg.inv(M),
                                           (img_original.shape[1],
                                            img_original.shape[0]))

    else:
        img_deskewed = None
        print(f'Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}')
        matchesMask = None

    return img_deskewed


def threshold_image(img):
    # TODO
    THRESHOLD = 0.85
    # THRESHOLD = 0.92
    img = cv2.threshold(img, thresh=np.round(THRESHOLD * 255),
                        maxval=1 * 255,
                        type=cv2.THRESH_BINARY_INV)[1]
    return img


def and_image(img1, img2):
    return np.round((img1 // 255 * img2 // 255) * 255)


def stack_image(img):
    img_cropped_1 = img[760:1400, 171:360]
    img_cropped_2 = img[760:1400, 500:689]
    img_cropped_3 = img[760:1400, 829:1018]

    img_stacked = np.vstack([img_cropped_1, img_cropped_2, img_cropped_3])
    # img_stacked = np.hstack([img_cropped_1, img_cropped_2, img_cropped_3])
    return img_stacked

def stack_image_2(img):
    # img = img_min_thresholded
    img_cropped_1 = img[776:1382 + 6, 116:360]
    img_cropped_2 = img[776:1382 + 3, 445:689]
    img_cropped_3 = img[776:1382 + 0, 774:1018]

    img_stacked = np.vstack([img_cropped_1, img_cropped_2, img_cropped_3])
    # plt.show(plt.imshow(img_stacked, 'gray'))
    # img_stacked = np.hstack([img_cropped_1, img_cropped_2, img_cropped_3])
    return img_stacked


def erode_image(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img = cv2.erode(img, kernel)
    return img


def dilate_image(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img = cv2.dilate(img, kernel)
    return img


def show_highlighted_circles(img, pairs=None):
    if not pairs:
        pairs = extract_circles(img)
    img = img.copy()
    for p in pairs:
        x1, y1, x2, y2 = p
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    show_image(img, unstack=True)


def extract_circles(img):
    # find all the 'white' shapes in the image
    lower = np.array(255)
    upper = np.array(255)
    shape_mask = cv2.inRange(img, lower, upper)

    # find the contours in the mask
    (_, cnts, _) = cv2.findContours(
        shape_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    pairs = []
    for c in cnts:
        c = c[:, 0, :]
        margin = 5
        x1 = c.min(axis=0)[0] - margin
        y1 = c.min(axis=0)[1] - margin
        x2 = c.max(axis=0)[0] + margin
        y2 = c.max(axis=0)[1] + margin
        pairs.append((x1, y1, x2, y2))

    return pairs


def count_circles(img, debug=False):
    pairs = extract_circles(img)
    if debug:
        print(f'Found {len(pairs)} black shapes')
        show_highlighted_circles(img, pairs)

    return len(pairs)


def remove_invalid_answers(img, debug=False):
    img = img.copy()

    heights = np.array([])
    invalid_answers = []
    spacing_threshold = 20

    pairs = extract_circles(img)
    for i, p in enumerate(pairs):
        x1, y1, x2, y2 = p
        new_height = (y1 + y2) / 2
        spacings = np.abs(heights - new_height)
        if spacings.size > 0 and spacings.min() < spacing_threshold:
            invalid_answers.append(i)
            invalid_answers.append(spacings.argmin())
        heights = np.concatenate([heights, [new_height]])

    for i in invalid_answers:
        x1, y1, x2, y2 = pairs[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    if debug:
        print(f'Found {len(pairs)} black shapes')
        pairs = extract_circles(img)
        show_highlighted_circles(img, pairs)

    return img
