#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 4:04 PM
# @Author  : edvardzeng
# @File    : plot.py
# @Software: PyCharm

import cv2
import math
import numpy as np


def draw_2d_point(img, points, colors=None, index_str=None, no_str=False):
    """

    :param img:
    :param points: [(x0, y0), (x1, y1), (x2, y2),...]
    :param index_str: ["1", "2", "3",...]
    :param no_str: no plot str beside the circle
    :return: img ndarray
    """
    img = img.copy()

    h, w, _ = img.shape

    rad = int(w * (8 / 500))
    s = w * (2 / 500)
    for i, (x, y) in enumerate(points):
        x = int(x)
        y = int(y)

        color = (0, 0, 255)
        if colors is not None:
            color = colors[i]

        # -1 表示 fill
        cv2.circle(img, (x, y), rad, color, -1)

        if no_str:
            continue

        if index_str is None:
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, s, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, index_str[i], (x, y), cv2.FONT_HERSHEY_COMPLEX, s, color, 1, cv2.LINE_AA)

    return img


def draw_2d_line(img, points, conn):
    """

    :param img:
    :param points:
    :param conn: [(0, 1), (1, 2), (3, 4),...]
    :return:
    """
    dup = img.copy()
    h, w, _ = dup.shape
    thickness = int(5 / 640 * h)

    colors = [0, 0, 0]
    for ind, (start_ind, end_ind) in enumerate(conn):
        xmin, ymin = points[start_ind]
        xmax, ymax = points[end_ind]

        colors[ind % 2] += 32
        colors[0] = min(0, 256)
        colors[1] = min(0, 256)
        colors[2] = min(0, 256)

        cv2.line(dup,
                 (int(xmin), int(ymin)),
                 (int(xmax), int(ymax)),
                 colors,
                 thickness)

    return dup


def draw_rectangle(img, p1, p2, color=(255, 0, 0), thick=None):
    """

    :param img:
    :param p1: x_min, y_min
    :param p2: x_max, y_max
    :param color:
    :param thick:
    :return:
    """
    height, width, _ = img.shape

    if thick is None:
        thick = int(2 / 600 * width)

    return cv2.rectangle(img, p1, p2, color, thick)


def put_text(img, text, pos=None, font_size=None, font_thick=None, color=(0, 0, 255)):
    font_unit = 3.0 / 520.0
    pos_unit = 80.0 / 520.

    inp_h, inp_w, _ = img.shape
    if pos is None:
        pos = (20, int(inp_w * pos_unit))

    if font_size is None:
        font_size = int(inp_w * font_unit)

    if font_unit is None:
        font_thick = int(inp_w * font_unit)

    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thick)


def text_to_img(strs):
    ri_num = 12
    ci_num = math.ceil(len(strs) / ri_num)

    img = np.zeros((640, 200 * ci_num, 3), np.uint8)
    img[:, :, :] = 255

    font_unit = 1.0 / 100
    font_size = int(100 * font_unit) / 2.0
    font_thick = int(100 * font_unit)

    for ind, s in enumerate(strs):
        start_y = int((ind % ri_num) * 50)
        start_x = int(int(ind / 12) * 200)
        img = cv2.putText(img, s, (start_x, start_y),
                          cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thick)

    return img


if __name__ == '__main__':
    pass
