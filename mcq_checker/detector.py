import cv2

from mcq_checker.utils.image import stack_image


def extract_answers(img):
    img = stack_image(img)
    img = cv2.bitwise_not(img)
    img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)[1]

    LENGTH = 41
    N_QUESTIONS = 45
    N_CHOICES = 4
    answers = dict([(i + 1, '?') for i in range(N_QUESTIONS)])
    for i in range(N_QUESTIONS):
        img_question = img[i * LENGTH: (i + 1) * LENGTH, 75:]

        d = []
        avgs = []
        for j in range(N_CHOICES):
            img_choice = img_question[:, LENGTH * j:LENGTH * (j + 1)]
            choice_avg = round(img_choice.sum() / (LENGTH ** 2 / 2))
            avgs.append(choice_avg)
            d.append((choice_avg, j))

        avg = (sum(avgs) - N_CHOICES * min(avgs)) / N_CHOICES

        d.sort(reverse=True)

        if d[0][0] > 1.05 * d[3][0] and (
                    d[0][0] - avg) > 1.5 * (d[1][0] - avg):
            ind = d[0][1]
            answers[i + 1] = 'ABCD'[ind]

    return answers
