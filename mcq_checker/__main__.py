import os
import re

import cv2
import pandas as pd

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
    remove_invalid_answers,
    show_highlighted_circles)

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
            # if i not in [13]:
            #  if i not in [4, 13, 15, 18, 39]:
            # if i not in [2, 17, 29, 46, 47, 48, 49, 50, 54, 56, 58, 60, 61, 62,
            #              63, 65, 66, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            #              80, 81, 82, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99,
            #              100, 101, 102, 103, 104, 105, 107, 109, 112, 114, 115,
            #              116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127,
            #              128, 130, 132, 134, 135, 136, 137, 138, 140, 141, 142,
            #              143, 144, 145, 146, 148, 150, 151, 152, 153, 154, 155,
            #              156, 157, 158, 159, 160, 161, 163, 167, 168, 169, 171,
            #              173, 176, 177, 179, 180, 181, 182, 183, 185, 186, 187,
            #              190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
            #              201, 202, 203, 204, 205, 206, 207, 208, 209, 212, 215,
            #              216, 219, 220, 223, 224, 225, 226, 227, 229, 232, 234,
            #              243, 245, 249, 256, 258, 262, 270, 275, 277, 278, 279,
            #              282]:
            # if i not in [17, 39, 54, 107, 122, 134, 142, 171, 245, 256]:
            #     continue

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
                errors.append(i)
                print(errors)
                # calculate_marks(img_model_filename, sample_file_path,
                #                 debug=True)

                # break
    except KeyboardInterrupt:
        pass

    print(errors)


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
    train()
    # test()
