import os
import cv2
import pandas as pd

from .img_processing import (
    show_image,
    load_image,
    deskew_image,
    threshold_image,
    and_image,
    stack_image,
    dilate_erode_image,
    detect_circles,
)

img_model = None
img_model_threshed = None


def calculate_marks(img_model_filename, img_sample_filename, debug=False):
    global img_model
    global img_model_threshed
    if debug:
        show = lambda x: show_image(x)
    else:
        show = lambda x: None

    if img_model is None:
        img_model = load_image(img_model_filename)
    img_sample = load_image(img_sample_filename)

    img_sample_deskewed = deskew_image(img_model, img_sample)

    show(img_model)
    show(img_sample_deskewed)

    if img_model_threshed is None:
        img_model_threshed = threshold_image(img_model)
    img_sample_threshed = threshold_image(img_sample_deskewed)

    show(img_model_threshed)
    show(img_sample_threshed)

    img_anded = and_image(img_model_threshed, img_sample_threshed)
    show(img_anded)

    img_stacked = stack_image(img_anded)
    show(img_stacked)

    img_dilated_eroded = dilate_erode_image(img_stacked)
    show(img_dilated_eroded)

    mark = detect_circles(img_dilated_eroded.copy(), debug=debug)
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

    total_abs_error = 0
    try:
        for t in train_set[['FileName', 'Mark']].itertuples():
            sample_file_path = f'data/dataset/train/{t.FileName}'
            expected_mark = t.Mark
            print(f'[{t.Index + 1:03}/{len(train_set):03}] '
                  f'{f"{t.Index / len(train_set) * 100:0.1f}":>5}%: '
                  f'{t.FileName:30}', end='', flush=True)

            output_mark = calculate_marks(img_model_filename, sample_file_path)

            output_dataframe.loc[t.Index] = [t.FileName, output_mark]
            save_csv(output_dataframe, output_csv_file_path)

            error = abs(output_mark - expected_mark)
            total_abs_error += error
            print(f'Output = {output_mark:02}, Expected = {expected_mark:02}:'
                  f' {f"Error = {error}" if error != 0 else "OK":14}'
                  f'Total Error = {total_abs_error:4}')

    except KeyboardInterrupt:
        pass


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
    test()
