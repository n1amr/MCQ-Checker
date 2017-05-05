import os

import cv2

from mcq_checker.constants import MODEL_ANSWERS, get_cached_image_path
from mcq_checker.deskewer import Deskewer
from mcq_checker.detector import extract_answers
from mcq_checker.utils.image import load_image


class Grader:
    def __init__(self, img_model_filename):
        self.img_model = load_image(img_model_filename)
        self.deskewer = Deskewer(self.img_model)

    def grade(self, img_path):
        cached_path = get_cached_image_path(img_path)
        if os.path.exists(cached_path):
            img = load_image(cached_path)
        else:
            img = load_image(img_path)
            img = self.deskewer.deskew(img)
            cv2.imwrite(cached_path, img)

        answers = extract_answers(img)

        marks = 0
        for i in range(1, 46):
            c = answers[i]
            if c is not None and c == MODEL_ANSWERS[i]:
                marks += 1

        return marks
