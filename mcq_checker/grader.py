import os
import re

import cv2

from mcq_checker.constants import MODEL_ANSWERS
from mcq_checker.naive_search import extract_answers
from mcq_checker.utils.image import load_image, deskew_image, \
    show_image


def get_cached_path(img_path):
    m = re.match(r'(?P<basename>.*)\.(?P<extension>\w+)', img_path)

    path = f"{m['basename']}_cached.{m['extension']}"
    return path


class Grader:
    def __init__(self, img_model_filename):
        self.img_model = load_image(img_model_filename)

    def grade(self, img_sample_filename, expected=None):
        cached_path = get_cached_path(img_sample_filename)
        if os.path.exists(cached_path):
            img_sample = load_image(cached_path)
        else:
            img_sample = load_image(img_sample_filename)
            img_sample = deskew_image(self.img_model, img_sample)
            cv2.imwrite(cached_path, img_sample)

        img = img_sample
        answers, marked_img = extract_answers(img)

        score = 0
        for i in range(1, 46):
            a = answers[i]
            if a is not None and a == MODEL_ANSWERS[i]:
                score += 1

        if expected is not None and score != expected:
            for i, ans in answers.items():
                print(i, ans)
            show_image(marked_img, complete=True)
            show_image(img)

        return score
