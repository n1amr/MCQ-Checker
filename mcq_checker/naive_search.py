import cv2

from mcq_checker.img_processing import stack_image_3


def extract_answers(img):
    img = stack_image_3(img)
    orig_img = img.copy()
    HEIGHT = 41
    answers = dict([(i, 0) for i in range(1, 46)])
    for i in range(45):
        img2 = img[i * HEIGHT: (i + 1) * HEIGHT, :]
        choices = []
        for j in range(4):
            img3 = img2[:, 75 + 40 * j:105 + 40 * j]
            img3 = cv2.bitwise_not(img3)
            img3 = cv2.threshold(img3, 50, 255, cv2.THRESH_TOZERO)[1]
            avg = img3.sum() / (25 * 23)
            choices.append(round(avg))
        avg = (sum(choices) - 4 * min(choices)) / len(choices)
        d = []
        for k in range(4):
            d.append((choices[k], k))
        d.sort(reverse=True)
        if d[0][0] / d[3][0] > 1.05 and (
                    d[0][0] - avg) > 1.5 * (d[1][0] - avg):
            ind = d[0][1]
            answers[i + 1] = 'ABCD'[ind]
            orig_img[i * HEIGHT: (i + 1) * HEIGHT,
            75 + 40 * ind:105 + 40 * ind] = 255

    return answers, orig_img
