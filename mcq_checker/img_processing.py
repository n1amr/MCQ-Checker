import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(img):
    plt.imshow(img, cmap='gray')
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
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
    # search_params = dict(checks=50)
    search_params = dict(checks=5)
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

        img_deskewed = cv2.warpPerspective(img_skewed, np.linalg.inv(M), (
            img_original.shape[1], img_original.shape[0]))

    else:
        img_deskewed = None
        print(f'Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}')
        matchesMask = None

    return img_deskewed


def threshold_image(img):
    return cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)[1]


def and_image(img1, img2):
    return np.round((img1 // 255 * img2 // 255) * 255)


def stack_image(img):
    img_cropped_1 = img[760:1400, 111:365]
    img_cropped_2 = img[760:1400, 428:682]
    img_cropped_3 = img[760:1400, 769:1023]

    img_stacked = np.vstack([img_cropped_1, img_cropped_2, img_cropped_3])
    return img_stacked


def dilate_erode_image(img):
    n = 11
    element = np.ones((n, n))

    img_dilated_eroded = cv2.dilate(cv2.erode(img, element), element)
    return img_dilated_eroded


def detect_circles(img, debug=False):
    # find all the 'white' shapes in the image
    lower = np.array(255)
    upper = np.array(255)
    shape_mask = cv2.inRange(img, lower, upper)

    # find the contours in the mask
    (_, cnts, _) = cv2.findContours(
        shape_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        print(f'I found {len(cnts)} black shapes')

        def show_highlighted_circles(img, cnts):
            for c in cnts:
                c = c[:, 0, :]
                margin = 5
                x1 = c.min(axis=0)[0] - margin
                y1 = c.min(axis=0)[1] - margin
                x2 = c.max(axis=0)[0] + margin
                y2 = c.max(axis=0)[1] + margin

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            show_image(img)

        show_highlighted_circles(img, cnts)

    return len(cnts)
