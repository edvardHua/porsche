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
    MODEL_HEAD = os.path.join(get_porsche_base_path(), "models/HeadSeg.onnx")
    MODEL_SKIN = os.path.join(get_porsche_base_path(), "models/GlobalSkinSeg.onnx")
    MODEL_PORTRAIT = os.path.join(get_porsche_base_path(), "models/PortraitSeg.onnx")

    def __init__(self, type="head", model_path=None):
        self.type = type
        if model_path is None:
            if type == "head":
                self.interpret = onnxruntime.InferenceSession(self.MODEL_HEAD)
            elif type == "skin":
                self.interpret = onnxruntime.InferenceSession(self.MODEL_SKIN)
            elif type == "portrait":
                self.interpret = onnxruntime.InferenceSession(self.MODEL_PORTRAIT)
        else:
            self.interpret = onnxruntime.InferenceSession(model_path)
        # 这里宽和高是相等的
        self.side_height, self.side_width = self.interpret.get_inputs()[0].shape[2:]

    def normalize_img(self, imgOri, scale, mean, val):
        img = np.array(imgOri.copy(), np.float32) / scale
        if len(img.shape) == 4:
            for j in range(img.shape[0]):
                for i in range(len(mean)):
                    img[j, :, :, i] = (img[j, :, :, i] - mean[i]) * val[i]
            return img
        else:
            for i in range(len(mean)):
                img[:, :, i] = (img[:, :, i] - mean[i]) * val[i]
            return img

    def infer(self, img, threshold=None, padding=True):
        ori_h, ori_w, _ = img.shape

        if padding:
            pad_img, w_offset, h_offset = padding_img(img)
        else:
            pad_img, w_offset, h_offset = img, 0, 0

        tmp_img = cv2.resize(pad_img, (self.side_width, self.side_height))

        if self.type == "portrait":
            inp_img = self.normalize_img(tmp_img, 1.0,
                                         [103.94, 116.78, 123.68],
                                         [0.017, 0.017, 0.017])
        else:
            inp_img = tmp_img / 127.5 - 1

        inp_img = inp_img.transpose((2, 0, 1))
        inp_img = inp_img[np.newaxis, :, :, :]
        inp_img = inp_img.astype(np.float32)

        pred = self.interpret.run(
            [self.interpret.get_outputs()[0].name],
            {self.interpret.get_inputs()[0].name: inp_img})[0]

        if self.type == "portrait":
            pred = pred[0, 1, ::]

        if threshold is not None:
            pred[pred < threshold] = 0.0
            pred[pred >= threshold] = 1.0
            pred = cv2.blur(pred, (3, 3))
            # pred[pred >= threshold] = 1.0

        if padding is False:
            return cv2.resize(pred, (ori_w, ori_h))

        restore_size_img = cv2.resize(pred[:, :], pad_img.shape[:2])
        res = restore_size_img[h_offset:(h_offset + ori_h), w_offset:(w_offset + ori_w)]

        return cv2.resize(res, (ori_w, ori_h))

    def __call__(self, frame, threshold=None, padding=True):
        return self.infer(frame, threshold, padding)

if __name__ == '__main__':
    se = Segmentation()
    img = cv2.imread("skin_res.jpg")
    print(img.shape)
    res = se.infer(img)
    print(res.shape)
