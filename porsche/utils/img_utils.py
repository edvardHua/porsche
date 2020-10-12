#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 8:03 PM
# @Author  : edvardzeng
# @File    : img_utils.py
# @Software: PyCharm

import os
import cv2
import numpy as np


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

    if dest_size[0] < ori_w and dest_size[1] < ori_h:
        raise Exception("The dest size must ")

    w_offset = max(0, int((dest_size[0] - ori_w) // 2))
    h_offset = max(0, int((dest_size[1] - ori_h) // 2))

    dest_img = cv2.copyMakeBorder(img, h_offset, h_offset, w_offset, w_offset, cv2.BORDER_CONSTANT,
                                  color)
    dest_img = cv2.resize(dest_img, (int(dest_size[0]), int(dest_size[1])))
    return (dest_img, w_offset, h_offset)


def cat_various_dims_imgs(imgs, display=False, out_path=None):
    height = 0
    width = 0

    if isinstance(imgs, list):
        for _ in imgs:
            h, w, c = _.shape
            height = max(h, height)
            width = max(w, width)
    else:
        height, width, _ = imgs.shape

    def nest_proc(img, pad=True):
        if width >= height:
            tmp_img = dynamic_ratio_resize(img, dest_height=height)
        else:
            tmp_img = dynamic_ratio_resize(img, dest_width=width)
        return padding_img(tmp_img, (width, height))[0] if pad else tmp_img

    if isinstance(imgs, list):
        all_imgs = [nest_proc(_) for _ in imgs]
        if width >= height:
            final_res = np.vstack(all_imgs)
        else:
            final_res = np.hstack(all_imgs)
    else:
        final_res = nest_proc(imgs, False)

    if display:
        cv2.imshow("res", final_res)
        cv2.waitKey(0)
        return

    if out_path:
        cv2.imwrite(out_path, final_res)
        return

    return final_res
