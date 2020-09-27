#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 4:04 PM
# @Author  : edvardzeng
# @File    : plot.py
# @Software: PyCharm

import cv2


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


if __name__ == '__main__':
    pass
