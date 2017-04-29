import os
import re

import cv2
import pandas as pd
import sys

from .img_processing import (
    show_image,
    load_image,
    deskew_image,
    threshold_image,
    and_image,
    stack_image,
    erode_image,
    dilate_image,
    count_circles,
    remove_invalid_answers
)
from .naive_search import extract_answers

img_model = None
img_model_threshed = None

MODEL_ANSWERS = {1: 'B', 2: 'C', 3: 'A', 4: 'A', 5: 'D',
                 6: 'A', 7: 'C', 8: 'C', 9: 'A', 10: 'C',
                 11: 'A', 12: 'B', 13: 'C', 14: 'C', 15: 'B',
                 16: 'A', 17: 'D', 18: 'B', 19: 'C', 20: 'B',
                 21: 'D', 22: 'C', 23: 'D', 24: 'B', 25: 'D',
                 26: 'C', 27: 'D', 28: 'D', 29: 'B', 30: 'C',
                 31: 'B', 32: 'B', 33: 'D', 34: 'C', 35: 'B',
                 36: 'C', 37: 'B', 38: 'C', 39: 'C', 40: 'A',
                 41: 'B', 42: 'B', 43: 'C', 44: 'C', 45: 'B', }


def get_cached_path(img_path):
    m = re.match(r'(?P<basename>.*)\.(?P<extension>\w+)', img_path)

    path = f"{m['basename']}_cached.{m['extension']}"
    return path


def calculate_marks(img_model_filename, img_sample_filename, expected=None):
    global img_model
    global img_model_threshed

    img_model = load_image(img_model_filename)

    cached_path = get_cached_path(img_sample_filename)
    if os.path.exists(cached_path):
        img_sample = load_image(cached_path)
    else:
        img_sample = load_image(img_sample_filename)
        img_sample = deskew_image(img_model, img_sample)
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


def load_csv(path, continue_=True):
    dataframe = pd.DataFrame(columns=['FileName', 'Mark'])
    if os.path.exists(path) and continue_:
        dataframe = pd.read_csv(path)
    return dataframe


def save_csv(dataframe, path):
    dataframe['Mark'] = dataframe['Mark'].astype('int8')
    dataframe.to_csv(path, index=False)


def print_errors(errors):
    for e in errors:
        print(f"{e['id']}:\t{e['mark']} != {e['expected']}\t{e['filename']}")


def train(samples=None):
    img_model_filename = 'data/model-answer.png'

    input_csv_file_path = 'data/train.csv'
    train_set = pd.read_csv(input_csv_file_path)

    output_csv_file_path = 'data/train-output.csv'
    output_dataframe = load_csv(output_csv_file_path, continue_=False)

    errors = []
    i = -1
    total_abs_error = 0
    try:
        for t in train_set[['FileName', 'Mark']].itertuples():
            i += 1
            if samples and (i not in samples):
                continue

            sample_file_path = f'data/dataset/train/{t.FileName}'
            expected_mark = t.Mark
            print(f'[{t.Index + 1:03}/{len(train_set):03}] '
                  f'{f"{t.Index / len(train_set) * 100:0.1f}":>5}%: '
                  f'{t.FileName:30}', end='', flush=True)

            output_mark = calculate_marks(img_model_filename, sample_file_path,
                                          expected=expected_mark)

            output_dataframe.loc[t.Index] = [t.FileName, output_mark]
            save_csv(output_dataframe, output_csv_file_path)

            error = output_mark - expected_mark
            total_abs_error += abs(error)
            print(f'Output = {output_mark:02}, Expected = {expected_mark:02}:'
                  f' {f"Error = {error}" if error != 0 else "OK":14}'
                  f'Total absolute error = {total_abs_error:4}')

            if error != 0:
                errors.append({'id': i, 'filename': sample_file_path,
                               'expected': expected_mark, 'mark': output_mark})
                print_errors(errors)
    except KeyboardInterrupt:
        pass

    print_errors(errors)


def test():
    img_model_filename = 'data/model-answer.png'

    input_csv_file_path = 'data/test.csv'
    test_set = pd.read_csv(input_csv_file_path)

    output_csv_file_path = 'data/test-output.csv'
    output_dataframe = load_csv(output_csv_file_path, continue_=False)

    try:
        for t in test_set[['FileName']].itertuples():
            sample_file_path = f'data/dataset/test/{t.FileName}'
            print(f'[{t.Index + 1:03}/{len(test_set):03}] '
                  f'{f"{t.Index / len(test_set) * 100:0.1f}":>5}%: '
                  f'{t.FileName:30}', end='', flush=True)

            output_mark = calculate_marks(img_model_filename, sample_file_path)

            output_dataframe.loc[t.Index] = [t.FileName, output_mark]
            save_csv(output_dataframe, output_csv_file_path)

            print(f'Output = {output_mark:02}:')

    except KeyboardInterrupt:
        pass


def main(*argv):
    samples = [15, 39, 44, 45, 50, 53, 54, 68, 71, 80, 107, 118, 121, 122, 133,
               149, 164, 169, 229, 245, 249, 251, 269, ]
    samples.sort()
    # train(samples)
    # train()
    test()

    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv))
