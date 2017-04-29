import cv2
import numpy as np

from mcq_checker.img_processing import show_image


class TemplateMatcher:
    def __init__(self):
        pass

    def reject(self, img):
        marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i, template in enumerate([*Template.REJECT, Template.EMPTY]):
            img_t = template.img
            w, h = img_t.shape[::-1]

            threshold = 0.80
            res = cv2.matchTemplate(img, img_t, cv2.TM_CCOEFF_NORMED)
            res[res < threshold] = 0
            res[res >= threshold] = 1
            loc = np.where(res >= threshold)

            points = [*zip(*loc[::-1])]
            if not len(points):
                continue
            min_points = [points[0]]
            for pt in points:
                if abs(pt[1] - min_points[-1][1]) > 20:
                    min_points.append(pt)

            for pt in min_points:
                pt_from = (pt[0], pt[1])
                pt_to = (pt[0] + w, pt[1] + h)
                cv2.rectangle(marked_img, pt_from, pt_to, (0, 0, 0), -1)

        return marked_img

    def marks_stacked(self, img):
        marked_img = self.reject(img)
        img = cv2.cvtColor(marked_img, cv2.COLOR_BGR2GRAY)

        answers = []
        min_points = []
        for i, template in enumerate(Template.TEMPLATES):
            img_t = template.img

            # threshold = 0.66
            threshold = 0.70
            res = cv2.matchTemplate(img, img_t, cv2.TM_CCOEFF_NORMED)
            res = local_maxima(res)

            loc = np.where(res >= threshold)

            points = [*zip(*loc[::-1])]
            if not len(points):
                continue

            pt = points[0]
            min_points.append((pt, i, res[pt[1], pt[0]]))
            for pt in points:
                if abs(pt[1] - min_points[-1][1]) > 20:
                    # min_points.append(pt)
                    min_points.append((pt, i, res[pt[1], pt[0]]))

        L_WIDTH = 2
        for pt, i, r in min_points:
            ans = Template.TEMPLATES[i].answer
            answers.append((pt[::-1], ans, r))

        HEIGHT = 40
        d_ans = {}
        d_ans_2 = {}
        for i in range(1, 47):
            d_ans[i] = set()
            d_ans_2[i] = set()
        answers = [*sorted(answers)]
        # min_answers = [answers[0]]
        last_ind = 0
        last_y = -50
        for pt, i, r in answers:
            y = pt[0]
            if (y - last_y) > 20:
                last_ind += (y - last_y) // 38

            ind = pt[0] // HEIGHT + 1
            d_ans[ind].add((i, r))
            d_ans_2[last_ind].add((i, r))

            last_y = y

        for ind in d_ans_2.keys():
            best_ans = None
            best_r = 0
            for ans, r in d_ans_2[ind]:
                if r > best_r:
                    best_r = r
                    best_ans = ans
            d_ans_2[ind] = (best_ans, best_r)

        w, h = img_t.shape[::-1]
        for ind in d_ans_2:
            best_ans = d_ans_2[ind][0]
            if best_ans is None:
                continue
            i = 'ABCD'.index(best_ans)
            pt = (10, (ind - 1) * 81 // 2)
            color = Template.TEMPLATES[i].color
            pt_from = (pt[0] + i * L_WIDTH, pt[1] + i * L_WIDTH)
            pt_to = (pt[0] + w, pt[1] + h)
            cv2.rectangle(marked_img, pt_from, pt_to, color, L_WIDTH)

        # show_image(marked_img, unstack=1)

        # print(min_answers)
        # print(d_ans)
        # t = True
        # for n in sorted(d_ans):
        #     t = t and d_ans[n] == d_ans_2[n]
        #     # print(n, d_ans[n], d_ans_2[n], d_ans[n] == d_ans_2[n], t)
        #     # assert d_ans[n] == d_ans_2[n]
        # if not t:
        #     show_image(marked_img, unstack=True)
        # # assert t
        return marked_img, d_ans_2


def local_maxima(img):
    x = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            l = [img[i, j],
                 img[i - 1, j],
                 img[i + 1, j],
                 img[i, j - 1],
                 img[i, j + 1],
                 img[i + 1, j + 1],
                 img[i + 1, j - 1],
                 img[i - 1, j + 1],
                 img[i - 1, j - 1], ]

            if img[i, j] == max(l):
                x[i, j] = img[i, j]
    return x


class Answer():
    ANSWER_A = 'A'
    ANSWER_B = 'B'
    ANSWER_C = 'C'
    ANSWER_D = 'D'


class Template():
    def __init__(self, answer, color,
                 template_file_path=None, template_img=None):
        if template_file_path is not None:
            self.img = cv2.imread(template_file_path, cv2.IMREAD_GRAYSCALE)
            self.img = cv2.threshold(self.img, 0.5, 1, cv2.THRESH_BINARY_INV)[1]
        elif template_img is not None:
            self.img = template_img
        else:
            raise ValueError('No img or path')
        self.answer = answer
        self.color = color


Template.TEMPLATE_A = Template(Answer.ANSWER_A, (255, 0, 255),
                               template_file_path='data/templates/template_a_2.png')
Template.TEMPLATE_B = Template(Answer.ANSWER_B, (0, 0, 255),
                               template_file_path='data/templates/template_b_2.png')
Template.TEMPLATE_C = Template(Answer.ANSWER_C, (0, 255, 0),
                               template_file_path='data/templates/template_c_2.png')
Template.TEMPLATE_D = Template(Answer.ANSWER_D, (255, 0, 0),
                               template_file_path='data/templates/template_d_2.png')
Template.TEMPLATES = [
    Template.TEMPLATE_A,
    Template.TEMPLATE_B,
    Template.TEMPLATE_C,
    Template.TEMPLATE_D,
]
img = Template.TEMPLATE_B.img
# img = cv2.bitwise_or(img, Template.TEMPLATE_A.img)
img = cv2.bitwise_and(img, Template.TEMPLATE_B.img)
# img = cv2.bitwise_or(img, Template.TEMPLATE_C.img)
img = cv2.bitwise_and(img, Template.TEMPLATE_D.img)

img = Template.TEMPLATE_A.img
img = cv2.bitwise_and(img, Template.TEMPLATE_B.img)
Template.EMPTY = Template('emp', (255, 255, 255), template_img=img)

Template.REJECT = []
for i in range(16):
    if i in [1, 2, 4, 8]:
        continue

    img = Template.EMPTY.img
    if i & (1 << 0):
        img = cv2.bitwise_or(img, Template.TEMPLATE_A.img)
    if i & (1 << 1):
        img = cv2.bitwise_or(img, Template.TEMPLATE_B.img)
    if i & (1 << 2):
        img = cv2.bitwise_or(img, Template.TEMPLATE_C.img)
    if i & (1 << 3):
        img = cv2.bitwise_or(img, Template.TEMPLATE_D.img)

    template = Template(f'inv({i})', (255, 255, 255), template_img=img)
    Template.REJECT.append(template)
