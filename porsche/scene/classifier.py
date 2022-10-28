# -*- coding: utf-8 -*-
# @Time : 2022/10/28 11:39
# @Author : zihua.zeng
# @File : classifier.py

import cv2
import os
import onnxruntime

import numpy as np

from porsche.utils.army_knife import get_porsche_base_path


class LightEnvClassifier:
    MODEL_PATH = "models/light_luminance_bs1_0831.onnx"

    def __init__(self):
        self.interpret = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), self.MODEL_PATH))

    def __call__(self, frame):
        # frame: cv2, image, 0-1, rgb
        # bright, dark
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, _1, inp_h, inp_w = self.interpret._sess.inputs_meta[0].shape
        frame = (cv2.resize(frame, (inp_w, inp_h))).astype(np.float32)
        frame /= 255.
        frame = frame.transpose((2, 0, 1))
        frame = frame[np.newaxis, :, :, :]
        pred = self.interpret.run(
            [self.interpret.get_outputs()[0].name],
            {self.interpret.get_inputs()[0].name: frame})[0]
        return pred
