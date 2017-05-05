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


def rotate_image(img, angle):
    rotation_mat = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    img = cv2.warpAffine(img, rotation_mat, img.T.shape)
    return img
