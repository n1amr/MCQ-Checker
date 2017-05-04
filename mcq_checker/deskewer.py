import math

import cv2
import numpy as np

from mcq_checker.utils.image import rotate_image


class Deskewer:
    def __init__(self, img_model):
        self.img_model = Deskewer.horizontal_img(img_model)
        self.model_centers = Deskewer.detect_alignment_circles(self.img_model)

    def deskew(self, img):
        img = Deskewer.horizontal_img(img)

        sample_centers = Deskewer.detect_alignment_circles(img)

        shift_x = int(((self.model_centers[0][0] - sample_centers[0][0]) +
                       (self.model_centers[1][0] - sample_centers[1][0])) / 2)
        shift_y = int(((self.model_centers[0][1] - sample_centers[0][1]) +
                       (self.model_centers[1][1] - sample_centers[1][1])) / 2)

        if shift_y < 0:
            img = img[abs(shift_y):, :]
        else:
            img = np.vstack([np.zeros((shift_y, img.shape[1])),
                             img])

        if shift_x < 0:
            img = img[:, abs(shift_x):]
        else:
            img = np.hstack([np.zeros((img.shape[0], shift_x)),
                             img])

        img = img.astype('uint8')
        return img

    @staticmethod
    def detect_alignment_circles(img):
        img = cv2.threshold(img.copy(), 150, 255, cv2.THRESH_BINARY_INV)[1]
        lower = np.array(255)
        upper = np.array(255)
        shape_mask = cv2.inRange(img, lower, upper)

        # find the contours in the mask
        (_, contours, _) = cv2.findContours(
            shape_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for c in contours:
            c = c[:, 0, :]
            margin = 5
            x1 = c.min(axis=0)[0] - margin
            y1 = c.min(axis=0)[1] - margin
            x2 = c.max(axis=0)[0] + margin
            y2 = c.max(axis=0)[1] + margin
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height
            segment = img[y1:y2, x1:x2]
            if (6000 < area < 10000
                and 0.7 * height < width < height * 1.4
                and y1 > img.shape[0] * 2 / 3
                and segment.mean() > 120):
                centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

        assert len(centers) == 2

        centers.sort()
        return centers

    @staticmethod
    def horizontal_img(img):
        centers = Deskewer.detect_alignment_circles(img)
        angle = Deskewer.calc_rotation_angle(centers)
        return rotate_image(img, angle)

    @staticmethod
    def calc_rotation_angle(centers):
        angle = math.atan2(centers[1][1] - centers[0][1],
                           centers[1][0] - centers[0][0])
        return angle / math.pi * 180
