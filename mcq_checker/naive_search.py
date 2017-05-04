import cv2

from mcq_checker.utils.image import stack_image


def extract_answers(img):
    # img_stacked = np.vstack([
    #     img[776:1392 - 15, 116:360],
    #     img[776:1392 - 15, 445:689],
    #     img[776:1392 - 15, 774:1018]])
    # show_image(img_stacked , complete=True)
    # show_image(stack_image(img), complete=True)
    img = stack_image(img)
    # show_image(img)

    HEIGHT = 41
    N_QUESTIONS = 45
    answers = dict([(i + 1, '?') for i in range(N_QUESTIONS)])
    for i in range(N_QUESTIONS):
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
        if d[0][0] > 1.05 * d[3][0] and (
                    d[0][0] - avg) > 1.5 * (d[1][0] - avg):
            ind = d[0][1]
            answers[i + 1] = 'ABCD'[ind]
            img[i * HEIGHT: (i + 1) * HEIGHT,
            75 + 40 * ind:105 + 40 * ind] = 255

    return answers, img
