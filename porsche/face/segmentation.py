#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/13 4:30 PM
# @Author  : edvardzeng
# @File    : segmentation.py
# @Software: PyCharm


import os
import cv2
import numpy as np
import onnxruntime
from porsche.utils.img_utils import padding_img
from porsche.utils.army_knife import get_porsche_base_path


class Segmentation:

    def __init__(self, type="head"):
        if type == "head":
            self.interpret = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), "models/HeadSeg.onnx"))
        else:
            self.interpret = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), "models/SkinSeg.onnx"))
        # 这里宽和高是相等的
        self.side_len = self.interpret.get_inputs()[0].shape[2]

    def infer(self, img, threshold=None):
        ori_h, ori_w, _ = img.shape
        pad_img, w_offset, h_offset = padding_img(img)

        # scale = pad_img.shape[0] / self.side_len

        tmp_img = cv2.resize(pad_img, (self.side_len, self.side_len))

        inp_img = tmp_img / 127.5 - 1
        inp_img = inp_img.transpose((2, 0, 1))
        inp_img = inp_img[np.newaxis, :, :, :]

        inp_img = inp_img.astype(np.float32)
        pred = self.interpret.run(
            [self.interpret.get_outputs()[0].name],
            {self.interpret.get_inputs()[0].name: inp_img})[0]

        if threshold is not None:
            pred[pred < threshold] = 0.0
            pred[pred >= threshold] = 1.0
            pred = cv2.blur(pred, (3, 3))
            # pred[pred >= threshold] = 1.0

        restore_size_img = cv2.resize(pred[:, :], pad_img.shape[:2])
        res = restore_size_img[h_offset:(h_offset + ori_h), w_offset:(w_offset + ori_w)]

        return cv2.resize(res, (ori_w, ori_h))


if __name__ == '__main__':
    se = Segmentation()
    img = cv2.imread("skin_res.jpg")
    print(img.shape)
    res = se.infer(img)
    print(res.shape)
