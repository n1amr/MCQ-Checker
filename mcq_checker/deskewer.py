import math

import cv2
import numpy as np


class Deskewer:
    def __init__(self, img_model):
        self.img_model = self.horizontal_img(img_model)
        self.model_centers = self.extract_big_circles(self.img_model)
        surf = cv2.xfeatures2d.SURF_create(400)
        self.kp1, self.des1 = surf.detectAndCompute(self.img_model, None)

    def deskew(self, img):
        img = self.horizontal_img(img)

        big_circles_cntrs = self.extract_big_circles(img)

        shift_x = ((self.model_centers[0][0] - big_circles_cntrs[0][0]) +
                   (self.model_centers[1][0] - big_circles_cntrs[1][0])) / 2
        shift_y = ((self.model_centers[0][1] - big_circles_cntrs[0][1]) +
                   (self.model_centers[1][1] - big_circles_cntrs[1][1])) / 2

        shift_y = int(shift_y)
        shift_x = int(shift_x)

        if shift_y < 0:
            img = img[-int(shift_y):, :]
        else:
            img = np.vstack([
                np.zeros((shift_y, img.shape[1])),
                img
            ])

        if shift_x < 0:
            img = img[:, -int(shift_x):]
        else:
            img = np.hstack([
                np.zeros((img.shape[0], shift_x)),
                img
            ])

        img = img.astype('uint8')
        return img

    def extract_big_circles(self, img):
        img = cv2.threshold(img.copy(), 150, 255, cv2.THRESH_BINARY_INV)[1]
        lower = np.array(255)
        upper = np.array(255)
        shape_mask = cv2.inRange(img, lower, upper)

        # find the contours in the mask
        (_, cnts, _) = cv2.findContours(
            shape_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for c in cnts:
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

    def horizontal_img(self, img):
        centers = self.extract_big_circles(img)
        angle = self.get_rotation_angle(centers)
        rot_img = self.rotate_image(img, angle)
        return rot_img

    def rotate_image(self, img, angle):
        rot = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        rot_img = cv2.warpAffine(img, rot, img.T.shape)
        return rot_img

    def get_rotation_angle(self, centers):
        ang = math.atan2(centers[1][1] - centers[0][1],
                         centers[1][0] - centers[0][0])
        return ang / math.pi * 180
