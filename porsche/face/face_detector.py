#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 11:10 AM
# @Author  : edvardzeng
# @File    : face_detector.py
# @Software: PyCharm

import cv2
import os
import csv
import numpy as np
import onnxruntime
from porsche.utils.army_knife import get_porsche_base_path


class FaceDetector():
    r"""

    output head bbox: [xmin, ymin, xmax, ymax]
           heat keypt: [[x0, y0]...[x5, y5]]
    """

    MODEL_PATH = "models/blazeface.onnx"
    ANCHOR_PATH = "models/blazeface_anchors.csv"

    def __init__(self, score_threshold=0.6, iou_threshold=0.6, need_expand=False):
        self.interp_face = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), self.MODEL_PATH))
        self.input_name = self.interp_face.get_inputs()[0].name
        self.output_name = self.interp_face.get_outputs()[0].name
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.need_expand = need_expand
        with open(os.path.join(get_porsche_base_path(), self.ANCHOR_PATH), "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]

    @staticmethod
    def _im_normalize(img):
        return np.ascontiguousarray(
            2 * ((img / 255) - 0.5).astype('float32'))

    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0, 0), (0, 1)), constant_values=1, mode='constant')

    @staticmethod
    def _sim(box0, box1, do_iou=True):
        x0, y0, w0, h0 = box0
        x1, y1, w1, h1 = box1
        area0 = w0 * h0
        area1 = w1 * h1
        xmin = max(x0 - w0 / 2, x1 - w1 / 2)
        ymin = max(y0 - h0 / 2, y1 - h1 / 2)
        xmax = min(x0 + w0 / 2, x1 + w1 / 2)
        ymax = min(y0 + h0 / 2, y1 + h1 / 2)
        i = max(0, xmax - xmin) * max(0, ymax - ymin)
        if do_iou:
            u = area0 + area1 - i
        else:
            # modified jaccard
            u = area1

        return i / (u + 1e-6)

    def non_maximum_suppression(self, reg, anchors, scores,
                                weighted=True, max_results=-1):

        sorted_idxs = scores.argsort()[::-1].tolist()

        abs_reg = np.copy(reg)

        # turn relative bbox/keyp into absolute bbox/keyp
        for idx in sorted_idxs:
            center = anchors[idx, :2] * 128
            for j in range(2):
                abs_reg[idx, j] = center[j] + abs_reg[idx, j]
                abs_reg[idx, (j + 4)::2] = center[j] + abs_reg[idx, (j + 4)::2]

        remain_idxs = sorted_idxs
        output_regs = abs_reg[0:0, :]

        while len(remain_idxs) > 0:
            # separate remain_idxs into candids and remain
            candids = []
            remains = []
            idx0 = remain_idxs[0]
            for idx in remain_idxs:
                sim = self._sim(abs_reg[idx0, :4], abs_reg[idx, :4])
                if sim >= self.iou_threshold:
                    candids.append(idx)
                else:
                    remains.append(idx)

            # compute weighted bbox/keyp
            if not weighted:
                weighted_reg = abs_reg[idx0, :]
            else:
                weighted_reg = 0
                total_score = 0
                for idx in candids:
                    total_score += scores[idx]
                    weighted_reg += scores[idx] * abs_reg[idx, :]
                weighted_reg /= total_score

            # add a new instance
            output_regs = np.concatenate((output_regs, weighted_reg.reshape(1, -1)), axis=0)

            remain_idxs = remains

            if max_results > 0 and output_regs.shape[0] >= max_results:
                break

        return output_regs

    def detect_face(self, img_norm):
        assert -1 <= img_norm.min() and img_norm.max() <= 1, \
            "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (128, 128, 3), \
            "img_norm shape must be (128, 128, 3)"

        res = self.interp_face.run([self.output_name], {self.input_name: img_norm[np.newaxis, :, :, :]})[0]

        out_reg = res[0, :, :16]
        out_clf = res[0, :, 16]
        # print(out_reg.shape, out_clf.shape)
        out_scr = self._sigm(out_clf)

        ids = []
        for i in range(896):
            if out_scr[i] > self.score_threshold:
                ids.append(i)

        # finding the best prediction
        detection_mask = out_scr > self.score_threshold
        filtered_detect = out_reg[detection_mask]
        filtered_anchors = self.anchors[detection_mask]
        filtered_scores = out_scr[detection_mask]

        if filtered_detect.shape[0] == 0:
            print("No faces found")
            return None, None

        # perform non-maximum suppression
        candidate_detect = self.non_maximum_suppression(filtered_detect, filtered_anchors, filtered_scores)

        bboxs = []
        keyps = []

        for idx in range(candidate_detect.shape[0]):
            # bounding box center offsets, width and height
            bbox = candidate_detect[idx, :4]

            # 6 initial keypoints
            keyp = candidate_detect[idx, 4:].reshape(-1, 2)

            bboxs.append(bbox)
            keyps.append(keyp)

        return bboxs, keyps

    def preprocess_img(self, img):
        # fit the image into a 128x128 square
        shape = np.r_[img.shape]
        pad_all = (shape.max() - shape[:2]).astype('uint32')
        pad = pad_all // 2
        img_pad = np.pad(
            img,
            ((pad[0], pad_all[0] - pad[0]), (pad[1], pad_all[1] - pad[1]), (0, 0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (128, 128))
        img_small = np.ascontiguousarray(img_small)

        # bgr -> rgb
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad

    def __call__(self, img):
        """
        bboxes [] = [[xmin, ymin, xmax, ymax],...]
        keypts [] = [[x, y], [x1, y1],...]

        :param img:
        :return: bboxes [], keypts []
        """
        img = img.copy()
        ori_h, ori_w, _ = img.shape
        img_pad, img_norm, pad = self.preprocess_img(img)

        bboxs, keyps = self.detect_face(img_norm)
        if bboxs is None:
            return [], []

        scale = max(img.shape) / 128

        list_keyp = []
        list_bbox = []
        for i in range(len(bboxs)):
            bbox = bboxs[i]
            keyp = keyps[i]

            bbox *= scale
            keyp *= scale
            bbox[:2] -= pad[::-1]
            keyp -= pad[::-1]

            # xcenter, ycenter, width, height => xmin, ymin, xmax, ymax
            if self.need_expand:
                xmin = max(0, bbox[0] - (bbox[2] / 2) * 1.5)
                ymin = max(0, bbox[1] - (bbox[3] / 2) * 2.5)

                xmax = min(ori_w, bbox[0] + (bbox[2] / 2) * 1.5)
                ymax = min(ori_h, bbox[1] + (bbox[3] / 2) * 1.5)
                new_box = [int(xmin), int(ymin), int(xmax), int(ymax)]
            else:
                new_box = [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2),
                           int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]

            list_keyp.append(keyp)
            list_bbox.append(new_box)

        return list_bbox, list_keyp


if __name__ == '__main__':
    img = cv2.imread("assets/images/header1.png")
