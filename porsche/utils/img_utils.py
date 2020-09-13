#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 8:03 PM
# @Author  : edvardzeng
# @File    : img_utils.py
# @Software: PyCharm

import os
import cv2


def dynamic_ratio_resize(img, dest_width=None, dest_height=None):
    assert dest_width or dest_height
    ori_h, ori_w, _ = img.shape

    if dest_width is None:
        dest_width = (ori_w / ori_h) * dest_height
    else:
        dest_height = (ori_h / ori_w) * dest_width

    return cv2.resize(img, (int(dest_width), int(dest_height)))


def padding_img(img, dest_size=None, color=(255, 255, 255)):
    ori_h, ori_w, _ = img.shape

    if dest_size is None:
        if ori_h >= ori_w:
            dest_size = (ori_h, ori_h)
        else:
            dest_size = (ori_w, ori_w)

    if dest_size[0] < ori_w or dest_size[1] < ori_h:
        raise Exception("The dest size must ")

    w_offset = int((dest_size[0] - ori_w) // 2)
    h_offset = int((dest_size[1] - ori_h) // 2)

    dest_img = cv2.copyMakeBorder(img, h_offset, h_offset, w_offset, w_offset, cv2.BORDER_CONSTANT,
                                  color)
    dest_img = cv2.resize(dest_img, (int(dest_size[0]), int(dest_size[1])))
    return (dest_img, w_offset, h_offset)


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

    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, font_size, color, font_thick)
