import sys

import cv2


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
    sys.exit(main(*sys.argv))
