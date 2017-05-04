import os

import cv2

from mcq_checker.constants import MODEL_ANSWERS, get_cached_image_path
from mcq_checker.naive_search import extract_answers
from mcq_checker.utils.image import load_image, show_image
from mcq_checker.deskewer import Deskewer


class Grader:
    def __init__(self, img_model_filename):
        self.img_model = load_image(img_model_filename)
        self.deskewer = Deskewer(self.img_model)

    def grade(self, img_path, expected=None):
        cached_path = get_cached_image_path(img_path)
        if os.path.exists(cached_path):
            img = load_image(cached_path)
        else:
            img = load_image(img_path)
            img = self.deskewer.deskew(img)
            cv2.imwrite(cached_path, img)

        answers, marked_img = extract_answers(img)

        marks = 0
        for i in range(1, 46):
            c = answers[i]
            if c is not None and c == MODEL_ANSWERS[i]:
                marks += 1

        if expected is not None and marks != expected:
            for i, ans in answers.items():
                print(i, ans)
                # show_image(marked_img, complete=True)
                # show_image(img)

        return marks
