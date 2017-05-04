import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(img, msg=None, unstack=False, complete=True):
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


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def stack_image(img):
    img_stacked = np.vstack([
        img[776:1392, 116:360],
        img[776:1392, 445:689],
        img[776:1392, 774:1018]])
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
