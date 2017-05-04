import sys

import pandas as pd

from mcq_checker import constants
from mcq_checker.constants import get_train_image_path, get_test_image_path
from mcq_checker.grader import Grader
from mcq_checker.utils.csv import load_csv, save_csv


def print_errors(errors):
    for e in errors:
        print(f"{e['id']}:\t{e['mark']} != {e['expected']}\t{e['filename']}")


def train(grader, samples=None):
    train_set = pd.read_csv(constants.TRAIN_INPUT_CSV_FILE_PATH)
    output_dataframe = load_csv(constants.TRAIN_OUTPUT_CSV_FILE_PATH,
                                continue_=False)

    errors = []
    total_abs_error = 0

    i = -1
    for t in train_set[['FileName', 'Mark']].itertuples():
        i += 1
        if samples and (i not in samples):
            continue

        img_path = get_train_image_path(t.FileName)
        print(f'[{t.Index + 1:03}/{len(train_set):03}] '
              f'{f"{t.Index / len(train_set) * 100:0.1f}":>5}%: '
              f'{t.FileName:30}', end='', flush=True)

        expected = t.Mark
        output = grader.grade(img_path, expected=expected)

        output_dataframe.loc[t.Index] = [t.FileName, output]
        save_csv(constants.TRAIN_OUTPUT_CSV_FILE_PATH, output_dataframe)

        error = output - expected
        total_abs_error += abs(error)
        print(f'Marks = {output:02}, '
              f'Expected = {expected:02}:'
              f' {f"Error = {error}" if error != 0 else "OK":14}'
              f'Total absolute error = {total_abs_error:4}')

        if error != 0:
            errors.append({'id': i,
                           'filename': img_path,
                           'expected': expected,
                           'mark': output})

    print_errors(errors)


def test(grader):
    test_set = pd.read_csv(constants.TEST_INPUT_CSV_FILE_PATH)

    output_dataframe = load_csv(constants.TEST_OUTPUT_CSV_FILE_PATH,
                                continue_=False)

    for t in test_set[['FileName']].itertuples():
        sample_file_path = get_test_image_path(t.FileName)
        print(f'[{t.Index + 1:03}/{len(test_set):03}] '
              f'{f"{t.Index / len(test_set) * 100:0.1f}":>5}%: '
              f'{t.FileName:30}', end='', flush=True)

        output = grader.grade(sample_file_path)
        output_dataframe.loc[t.Index] = [t.FileName, output]

        print(f'Marks = {output:02}')
        save_csv(constants.TEST_OUTPUT_CSV_FILE_PATH, output_dataframe)


def print_usage():
    print('Usage: python -m mcq_checker <train [sample_numbers]|test>')


def main(*args):
    if len(args) < 2:
        print_usage()
        return 1

    grader = Grader(constants.IMG_MODEL_FILE_PATH)

    if args[1] == 'train':
        samples = [*map(int, args[2:])]
        samples.sort()
        train(grader, samples)
    elif args[1] == 'test':
        test(grader)
    else:
        print_usage()
        return 1

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(*sys.argv))
    except KeyboardInterrupt:
        pass
