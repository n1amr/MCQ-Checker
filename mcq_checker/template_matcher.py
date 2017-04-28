import cv2
import numpy as np


class TemplateMatcher:
    template_a = cv2.imread('data/templates/template_a_2.png', 0)
    template_b = cv2.imread('data/templates/template_b_2.png', 0)
    template_c = cv2.imread('data/templates/template_c_2.png', 0)
    template_d = cv2.imread('data/templates/template_d_2.png', 0)

    color_a = (0, 0, 0)
    color_b = (0, 0, 255)
    color_c = (0, 255, 0)
    color_d = (255, 0, 0)

    def __init__(self, img_gray):
        # img_rgb = cv2.imread('data/model-answer.png')
        # img_rgb = cv2.imread('data/dataset/train/S_8_hppscan138_cached.png')
        self.img_gray = img_gray
        self.img_rgb = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2BGR)

    # def marks(self):
    #     marked_img = self.img_rgb.copy()
    #     answers = []
    #     for template, color, ans in [(self.template_a, self.color_a, 'a'),
    #                                  (self.template_b, self.color_b, 'b'),
    #                                  (self.template_c, self.color_c, 'c'),
    #                                  (self.template_d, self.color_d, 'd'), ]:
    #         w, h = template.shape[::-1]
    #
    #         res = cv2.matchTemplate(self.img_gray, template,
    #                                 cv2.TM_CCOEFF_NORMED)
    #         threshold = 0.70
    #         loc = np.where(res >= threshold)
    #
    #         for pt in zip(*loc[::-1]):
    #             cv2.rectangle(marked_img, pt, (pt[0] + w, pt[1] + h), color, 4)
    #             answers.append((pt[::-1], ans))
    #
    #     answers = [*sorted(answers)]
    #     min_answers = [answers[0]]
    #     for ans in answers:
    #         if abs(min_answers[-1][0][1] - ans[0][1]) > 20:
    #             min_answers.append(ans)
    #
    #     return marked_img

    def marks_stacked(self, img):
        marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        answers = []
        for i, template in enumerate(Template.TEMPLATES):
            img_t = template.img
            color = template.color
            ans = template.answer
            w, h = img_t.shape[::-1]

            res = cv2.matchTemplate(img, img_t, cv2.TM_CCOEFF_NORMED)
            threshold = 0.66
            loc = np.where(res >= threshold)

            points = [*zip(*loc[::-1])]
            if not len(points):
                continue
            min_points = [points[0]]
            for pt in points:
                if abs(pt[1] - min_points[-1][1]) > 20:
                    min_points.append(pt)

            L_WIDTH = 2
            for pt in min_points:
                pt_from = (pt[0] + i * L_WIDTH, pt[1] + i * L_WIDTH)
                pt_to = (pt[0] + w, pt[1] + h)
                cv2.rectangle(marked_img, pt_from, pt_to, color, L_WIDTH)
                answers.append((pt[::-1], ans))

        def same_answer(ans1, ans2):
            HEIGHT = 20
            return (abs(ans2[0][0] - ans1[0][0]) < HEIGHT
                    and ans1[1] == ans2[1])

        HEIGHT = 40
        d_ans = {}
        d_ans_2 = {}
        for i in range(1, 47):
            d_ans[i] = set()
            d_ans_2[i] = set()
        answers = [*sorted(answers)]
        min_answers = [answers[0]]
        last_ind = 0
        last_y = -50
        for ans in answers:
            y = ans[0][0]
            if (y - last_y) > 20:
                last_ind += (y - last_y) // 38
            min_answers.append(ans)

            d_ans[ans[0][0] // HEIGHT + 1].add(ans[1])
            d_ans_2[last_ind].add(ans[1])

            last_y = y

        # print(min_answers)
        # print(d_ans)
        t = True
        for n in sorted(d_ans):
            t = t and d_ans[n] == d_ans_2[n]
            # print(n, d_ans[n], d_ans_2[n], d_ans[n] == d_ans_2[n], t)
            # assert d_ans[n] == d_ans_2[n]
        assert t
        return marked_img, d_ans_2


class Answer():
    # def __init__(self, name):
    #     self.name = name
    #
    # def __str__(self):
    #     return self.name

    ANSWER_A = 'A'
    ANSWER_B = 'B'
    ANSWER_C = 'C'
    ANSWER_D = 'D'


# Answer.ANSWER_A = Answer('A')
# Answer.ANSWER_B = Answer('B')
# Answer.ANSWER_C = Answer('C')
# Answer.ANSWER_D = Answer('D')


class Template():
    def __init__(self, template_file_path, answer, color):
        self.img = cv2.imread(template_file_path, cv2.IMREAD_GRAYSCALE)
        self.img = cv2.threshold(self.img, 0.5, 1, cv2.THRESH_BINARY_INV)[1]
        self.answer = answer
        self.color = color


Template.TEMPLATE_A = Template('data/templates/template_a_2.png',
                               Answer.ANSWER_A, (255, 0, 255))
Template.TEMPLATE_B = Template('data/templates/template_b_2.png',
                               Answer.ANSWER_B, (0, 0, 255))
Template.TEMPLATE_C = Template('data/templates/template_c_2.png',
                               Answer.ANSWER_C, (0, 255, 0))
Template.TEMPLATE_D = Template('data/templates/template_d_2.png',
                               Answer.ANSWER_D, (255, 0, 0))
Template.TEMPLATES = [
    Template.TEMPLATE_A,
    Template.TEMPLATE_B,
    Template.TEMPLATE_C,
    Template.TEMPLATE_D,
]
