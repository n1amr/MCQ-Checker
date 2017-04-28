import os
import re
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt

from mcq_checker.img_processing import stack_image_2
from mcq_checker.template_matcher import TemplateMatcher
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

img_model = None
img_model_threshed = None


def get_cached_path(img_path):
    m = re.match(r'(?P<basename>.*)\.(?P<extension>\w+)', img_path)

    path = f"{m['basename']}_cached.{m['extension']}"
    return path


def calculate_marks(img_model_filename, img_sample_filename, debug=False):
    global img_model
    global img_model_threshed
    show = show_image
    if not debug:
        def f(*args, **kwargs):
            pass

        show = f

    img_model = load_image(img_model_filename)
    # show(img_model, 'model: original')

    cached_path = get_cached_path(img_sample_filename)
    if os.path.exists(cached_path):
        img_sample = load_image(cached_path)
        show(img_sample, 'sample: loaded cached')
    else:
        img_sample = load_image(img_sample_filename)
        show(img_sample, 'sample: original')

        img_sample = deskew_image(img_model, img_sample)
        show(img_sample, 'sample: deskewed')

        cv2.imwrite(cached_path, img_sample)

    img_max = cv2.max(img_model, img_sample)
    img_max_thresholded = threshold_image(img_max)

    img_min = cv2.min(img_model, img_sample)
    img_min_thresholded = threshold_image(img_min)

    # img_min_stacked = stack_image(img_min_thresholded)
    # img_max_stacked = stack_image(img_max_thresholded)

    img_min_stacked = stack_image_2(img_min_thresholded)
    img_max_stacked = stack_image_2(img_max_thresholded)

    tm = TemplateMatcher(img_min_stacked)

    # img_model_thresholded = threshold_image(img_model)
    # img_model_stacked = stack_image_2(img_model_thresholded)
    # img_model_marked, answers = tm.marks_stacked(img_model_stacked)
    # show_image(img_model_marked, unstack=True)
    # print(answers)

    MODEL_ANSWERS = {1: {'B'}, 2: {'C'}, 3: {'A'}, 4: {'A'}, 5: {'D'},
                     6: {'A'}, 7: {'C'}, 8: {'C'}, 9: {'A'}, 10: {'C'},
                     11: {'A'}, 12: {'B'}, 13: {'C'}, 14: {'C'}, 15: {'B'},
                     16: {'A'}, 17: {'D'}, 18: {'B'}, 19: {'C'}, 20: {'B'},
                     21: {'D'}, 22: {'C'}, 23: {'D'}, 24: {'B'}, 25: {'D'},
                     26: {'C'}, 27: {'D'}, 28: {'D'}, 29: {'B'}, 30: {'C'},
                     31: {'B'}, 32: {'B'}, 33: {'D'}, 34: {'C'}, 35: {'B'},
                     36: {'C'}, 37: {'B'}, 38: {'C'}, 39: {'C'}, 40: {'A'},
                     41: {'B'}, 42: {'B'}, 43: {'C'}, 44: {'C'}, 45: {'B'},
                     46: set()}

    img = img_sample.copy()
    img = cv2.bitwise_not(img)
    img = stack_image_2(img)
    img, answers = tm.marks_stacked(img)
    # show_image(img, unstack=True)
    # print(answers)

    score = 0
    for i in range(1, 46):
        a = answers[i]
        if len(a) == 1 and a == MODEL_ANSWERS[i]:
            score += 1
    return score

    # show_image(img_sample_marked, unstack=True)

    # img_min_marked = tm.marks_stacked(img_min_stacked)
    # show_image(img_min_marked, unstack=True)

    # img_max_marked = tm.marks_stacked(img_max_stacked)
    # show_image(img_max_marked, unstack=True)

    # img_anded = cv2.bitwise_and(img_min_marked, img_max_marked)
    # show_image(img_anded, unstack=True)

    # show_image(np.hstack([
    #     img_min_marked,
    #     img_max_marked,
    #     img_anded
    # ]), unstack=True)
    #
    return 0

    M = 10
    N = 10
    img_min_ = dilate_image(erode_image(img_min_stacked, (M, M)), (N, N))
    img_max_ = dilate_image(erode_image(img_max_stacked, (M, M)), (N, N))

    # img_min_ = img_min_stacked
    # img_max_ = img_max_stacked

    # show_image(img_min_, complete=True)
    # show_image(img_max_, complete=True)

    img_min_ = remove_invalid_answers(img_min_, debug=True)
    img_max_ = remove_invalid_answers(img_max_, debug=True)

    img_anded = and_image(img_min_, img_max_)
    # show_image(img_anded)

    x = count_circles(img_anded)
    return x

    # import IPython
    # IPython.embed()
    # err

    img_model = threshold_image(img_model)
    # show(img_model, 'model: thresholded')


    img_model = dilate_image(img_model, (13, 13))
    # show(img_model, 'model: dilated 13x13')
    img_model = erode_image(img_model, (13, 13))
    # show(img_model, 'model: eroded 13x13')

    # find common
    img_sample_tmp = img_sample.copy()

    img_sample_tmp = threshold_image(img_sample_tmp)
    show(img_sample_tmp, 'sample: thresholded')

    img_sample_tmp = erode_image(img_sample_tmp, (5, 5))
    show(img_sample_tmp, 'sample: eroded 5x5')
    img_sample_tmp = dilate_image(img_sample_tmp, (5, 5))
    show(img_sample_tmp, 'sample: dilated 5x5')

    img_sample_tmp = dilate_image(img_sample_tmp, (13, 13))
    show(img_sample_tmp, 'sample: dilated 13x13')
    img_sample_tmp = erode_image(img_sample_tmp, (13, 13))
    show(img_sample_tmp, 'sample: eroded 13x13')

    img_common = and_image(img_model, img_sample_tmp)
    show(img_common, 'common: model && sample')

    img_common = dilate_image(img_common, (13, 13))
    show(img_common, 'common: dilated 13x13')
    img_common = erode_image(img_common, (13, 13))
    show(img_common, 'common: eroded 13x13')

    img_common = erode_image(img_common, (13, 13))
    show(img_common, 'common: eroded 13x13')
    img_common = dilate_image(img_common, (13, 13))
    show(img_common, 'common: dilated 13x13')

    img_common = stack_image(img_common)
    show(img_common, 'common: stacked', True)
    # show_highlighted_circles(img_common)

    # filter valid
    img_sample_tmp = img_sample.copy()
    img_sample_tmp = threshold_image(img_sample_tmp)
    show(img_sample_tmp, 'sample: thresholded')

    img_valid = img_sample_tmp
    # img_valid = dilate_image(img_valid, (7, 7))
    # show(img_valid, 'valid: dilated 7x7')
    # img_valid = erode_image(img_valid, (7, 7))
    # show(img_valid, 'valid: eroded 7x7')

    img_valid = erode_image(img_valid, (7, 7))
    show(img_valid, 'valid: eroded 7x7')
    img_valid = erode_image(img_valid, (7, 7))
    show(img_valid, 'valid: eroded 7x7')
    img_valid = dilate_image(img_valid, (14, 14))
    show(img_valid, 'valid: dilated 14x14')

    img_valid = stack_image(img_valid)
    show(img_valid, 'valid: stacked', True)

    img_valid = remove_invalid_answers(img_valid, debug=debug)
    show(img_valid, 'valid: removed invalid answers', True)

    img_correct = and_image(img_common, img_valid)
    show(img_correct, 'correct: valid && common', True)
    # show_highlighted_circles(img_correct)

    mark = count_circles(img_correct, debug=debug)
    return mark


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


def train():
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
            # TODO
            # if i not in [17, 39, 54, 107, 134, 142, 171, 245, 256, 26]:
            # if i not in [54]:
            # if i not in [1, 17, 71, 107, 122, 149, 245, 269]:
            #     continue
            # if i not in [15, 54, 107, 118, 229]:
            #     continue
            if i not in [
                15, 54, 107, 118, 229, 39, 45, 107, 122, 269,
            ]:
                continue

            sample_file_path = f'data/dataset/train/{t.FileName}'
            expected_mark = t.Mark
            print(f'[{t.Index + 1:03}/{len(train_set):03}] '
                  f'{f"{t.Index / len(train_set) * 100:0.1f}":>5}%: '
                  f'{t.FileName:30}', end='', flush=True)

            output_mark = calculate_marks(img_model_filename, sample_file_path)

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
                # calculate_marks(img_model_filename, sample_file_path,
                #                 debug=True)

                # break
    except KeyboardInterrupt:
        pass

    # print(errors)
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
    if len(argv) < 2:
        print('Please pass image path')
        return -1

    img = cv2.imread(argv[1])
    cv2.imshow('output image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Hello world! Initial setup is OK')
    return 0


if __name__ == '__main__':
    # sys.exit(main(*sys.argv))
    # train()
    test()
