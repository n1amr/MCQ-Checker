import sys

import pandas as pd

from mcq_checker.grader import Grader
from mcq_checker.utils.csv import load_csv, save_csv

img_model = None
img_model_threshed = None


def print_errors(errors):
    for e in errors:
        print(f"{e['id']}:\t{e['mark']} != {e['expected']}\t{e['filename']}")


def train(grader, samples=None):
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

            output_mark = grader.grade(sample_file_path,
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


def test(grader):
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

            output_mark = grader.grade(sample_file_path)

            output_dataframe.loc[t.Index] = [t.FileName, output_mark]
            save_csv(output_dataframe, output_csv_file_path)

            print(f'Output = {output_mark:02}:')

    except KeyboardInterrupt:
        pass


def print_usage():
    print('Usage: python -m mcq_checker <train [sample_numbers]|test>')


def main(*args):
    if len(args) < 2:
        print_usage()
        return 1

    img_model_filename = 'data/model-answer.png'
    grader = Grader(img_model_filename)

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
    sys.exit(main(*sys.argv))
