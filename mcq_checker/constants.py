IMG_MODEL_FILE_PATH = 'data/model-answer.png'

TRAIN_INPUT_CSV_FILE_PATH = 'data/train.csv'
TRAIN_OUTPUT_CSV_FILE_PATH = 'data/train-output.csv'

TEST_INPUT_CSV_FILE_PATH = 'data/test.csv'
TEST_OUTPUT_CSV_FILE_PATH = 'data/test-output.csv'

MODEL_ANSWERS = {1: 'B', 2: 'C', 3: 'A', 4: 'A', 5: 'D',
                 6: 'A', 7: 'C', 8: 'C', 9: 'A', 10: 'C',
                 11: 'A', 12: 'B', 13: 'C', 14: 'C', 15: 'B',
                 16: 'A', 17: 'D', 18: 'B', 19: 'C', 20: 'B',
                 21: 'D', 22: 'C', 23: 'D', 24: 'B', 25: 'D',
                 26: 'C', 27: 'D', 28: 'D', 29: 'B', 30: 'C',
                 31: 'B', 32: 'B', 33: 'D', 34: 'C', 35: 'B',
                 36: 'C', 37: 'B', 38: 'C', 39: 'C', 40: 'A',
                 41: 'B', 42: 'B', 43: 'C', 44: 'C', 45: 'B', }


def get_train_image_path(filename):
    return f'data/dataset/train/{filename}'


def get_test_image_path(filename):
    return f'data/dataset/test/{filename}'
