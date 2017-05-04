import math

import cv2
import numpy as np


class Deskewer:
    def __init__(self, img_model):
        self.img_model = img_model
        surf = cv2.xfeatures2d.SURF_create(400)
        self.kp1, self.des1 = surf.detectAndCompute(self.img_model, None)

    def deskew(self, img, debug=False):
        surf = cv2.xfeatures2d.SURF_create(400)

        kp2, des2 = surf.detectAndCompute(img, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_params = dict(checks=10)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.kp1[m.queryIdx].pt
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

            img_deskewed = cv2.warpPerspective(img,
                                               np.linalg.inv(M),
                                               (self.img_model.shape[1],
                                                self.img_model.shape[0]))

        else:
            img_deskewed = None
            print(
                f'Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}')

        return img_deskewed