#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 7:26 PM
# @Author  : edvardzeng
# @File    : face_cartoon.py
# @Software: PyCharm


import os
import cv2
import json
import onnxruntime
import numpy as np

from porsche.face.face_detector import FaceDetector
from porsche.face.segmentation import Segmentation
from porsche.utils.army_knife import get_porsche_base_path


class FaceCartoon:
    MODEL_PATH = "models/student_gan_0902.onnx"
    BATCH_PATH = "models/student_gan_0902_dynamic_axes.onnx"
    DEFAULT_LUT_PATH = "models/default_lut.png"

    def __init__(self, bg_lut_file=None, custom_bg_func=None, model_path=None):
        self.fd = FaceDetector(need_expand=True)
        self.sehead = Segmentation(type="head")
        self.seskin = Segmentation(type="skin")
        self.model_path = model_path
        if model_path is not None:
            self.cartoon = onnxruntime.InferenceSession(model_path)
        else:
            self.cartoon = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), self.MODEL_PATH))

        _, _1, self.inp_h, self.inp_w = self.cartoon._sess.inputs_meta[0].shape

        self.custom_bg_func = custom_bg_func
        if bg_lut_file is not None:
            self.lut_img = cv2.imread(bg_lut_file)
        else:
            self.lut_img = cv2.imread(self.DEFAULT_LUT_PATH)

        self.call_counter = 0

    def bg_lut(self, frame):
        dest_img = frame.copy()
        mapping_blocks = []
        mapping_blocks_append = mapping_blocks.append
        for ri in range(8):
            for ci in range(8):
                start_ri = 64 * ri
                start_ci = 64 * ci
                block = self.lut_img[start_ri:(start_ri + 64), start_ci:(start_ci + 64), :]
                mapping_blocks_append(block)

        height, width, _ = dest_img.shape
        dest_img = ((dest_img.astype(np.float32)) / 4).astype(np.uint8)
        for ri in range(height):
            for ci in range(width):
                b, g, r = dest_img[ri, ci, :]
                res = mapping_blocks[b][g, r, :]
                dest_img[ri, ci, :] = res

        return dest_img

    def batch_infer(self, frames):
        if self.model_path is None:
            self.cartoon = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), self.BATCH_PATH))

        inp_frames = []
        for f in frames:
            inp_frames.append(np.squeeze(self.input_preprocess(f)))

        inp_frames = np.array(inp_frames)
        cartoon_faces = self.cartoon.run(["output"], {"input": inp_frames})[0]

        b, c, h, w = cartoon_faces.shape

        ret_frame = []
        for ind in range(b):
            cartoon_face = cartoon_faces[ind, :, :, :]
            cartoon_face = np.squeeze(cartoon_face)
            cartoon_face = cartoon_face.transpose([1, 2, 0])
            cartoon_face = ((cartoon_face + 1.) / 2) * 255.0

            cartoon_face = cartoon_face.astype(np.uint8)
            cartoon_face = cartoon_face[:, :, ::-1]
            ori_h, ori_w, _ = frames[ind].shape
            ret_frame.append(cv2.resize(cartoon_face, (ori_w, ori_h), interpolation=cv2.INTER_LANCZOS4))
        return ret_frame

    def infer(self, frame):
        ori_h, ori_w, _ = frame.shape

        inp_img = self.input_preprocess(frame)
        cartoon_face = self.cartoon.run(["output"], {"input": inp_img})[0]
        cartoon_face = np.squeeze(cartoon_face)
        cartoon_face = cartoon_face.transpose([1, 2, 0])
        cartoon_face = ((cartoon_face + 1.) / 2) * 255.0

        cartoon_face = cartoon_face.astype(np.uint8)
        cartoon_face = cartoon_face[:, :, ::-1]
        return cv2.resize(cartoon_face, (ori_w, ori_h), interpolation=cv2.INTER_LANCZOS4)

    def input_preprocess(self, img):
        img = cv2.resize(img, (self.inp_w, self.inp_h), interpolation=cv2.INTER_LANCZOS4)
        nor = img / 255.0
        inp = (nor - 0.5) / 0.5
        inp = inp[:, :, ::-1]
        inp = inp.transpose([2, 0, 1])
        inp = inp.astype(np.float32)
        return inp[np.newaxis, :, :, :]

    def __find_edges_canny(self, image, edge_multiplier=1.0):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = min(int(200 * (1 / edge_multiplier)), 254)
        edges = cv2.Canny(image_gray, thresh, thresh)
        return edges

    def __suppress_edge_blobs(self, edges, size, thresh, inverse):
        kernel = np.ones((size, size), dtype=np.float32)
        counts = cv2.filter2D(edges / 255.0, -1, kernel)

        if inverse:
            mask = (counts < thresh)
        else:
            mask = (counts >= thresh)

        edges = np.copy(edges)
        edges[mask] = 0
        return edges

    def __blend_edges(self, image, image_edges):
        image_edges = 1.0 - (image_edges / 255.0)
        image_edges = np.tile(image_edges[..., np.newaxis], (1, 1, 3))
        return np.clip(
            np.round(image * image_edges),
            0.0, 255.0
        ).astype(np.uint8)

    def cartoon_bg(self, frame):
        img = frame.copy()
        edge = self.__find_edges_canny(img)
        edge = self.__suppress_edge_blobs(edge, 3, 8, False)
        edge = self.__suppress_edge_blobs(edge, 5, 3, True)

        spatial_window_radius = int(15 * 1.0)
        color_window_radius = int(40 * 1.0)
        mean_img = cv2.pyrMeanShiftFiltering(img, spatial_window_radius, color_window_radius)
        return self.__blend_edges(mean_img, edge)

    def set_config(self, seg=True, blend=True, proc_bg=True, detect_face=True, lut_on_cartoon_face=False,
                   only_seg=False, face_bbox_result=None):
        """
        预先设置好运行时设置而不是在 call 的时候传参设置
        :param seg:
        :param blend:
        :param proc_bg:
        :param detect_face:
        :param lut_on_cartoon_face:
        :param only_seg:
        :param face_bbox_result: [(x, y, w, h), ...]
        :return:
        """
        self.apply_seg = seg
        self.apply_blend = blend
        self.apply_proc_bg = proc_bg
        self.apply_detect_face = detect_face
        self.apply_lut_to_cartoon_face = lut_on_cartoon_face
        self.only_seg = only_seg
        self.face_bbox_result = face_bbox_result
        if face_bbox_result is not None:
            self.face_bbox_result = json.load(open(face_bbox_result))

    def get_gaussian_filter(self):
        heatmap = np.zeros((224, 224))
        variance = 200
        mul = 1.0
        height, width = heatmap.shape
        c_x, c_y = (112, 112)
        for x_p in range(width):
            for y_p in range(height):
                dist_sq = (x_p - c_x) * (x_p - c_x) + \
                          (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / variance / variance
                new_val = np.exp(-exponent) * mul
                new_val = min(1, max(0, new_val))
                heatmap[y_p, x_p] = new_val
        return heatmap

    def __call__(self, frame):
        frame = frame.copy()
        if self.apply_detect_face:
            if self.face_bbox_result is not None:
                box = self.face_bbox_result[self.call_counter]
                # x, y, w, h => xmin, ymin, xmax, ymax
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                crop_face = frame[box[1]:box[3], box[0]:box[2], :]
            else:
                bboxes, _ = self.fd(frame)
                if len(bboxes) != 0:
                    box = bboxes[0]
                    crop_face = frame[box[1]:box[3], box[0]:box[2], :]
        else:
            crop_face = frame

        if self.apply_seg:
            # skin include neck
            # mask_head = self.sehead(crop_face)
            # mask_skin = self.seskin(crop_face)
            # mask = (mask_head + mask_skin) / 2.0

            # 如果使用了高斯模糊核作为 seg，就不需要推断 sehead

            if self.only_seg:
                # mask = self.get_gaussian_filter()
                # mh, mw, _ = crop_face.shape
                # mask = cv2.resize(mask, (mw, mh))
                # mask_ori = self.sehead(crop_face)
                # mask = mask_ori * mask
                mask = self.sehead(crop_face)
                mask[mask >= 0.8] = 1.0
                mask = cv2.blur(mask, (5, 5))
            else:
                # only head
                mask = self.sehead(crop_face)
                mask[mask >= 0.3] = 1.0
                mask = cv2.blur(mask, (5, 5))
                crop_face = (crop_face * mask[:, :, np.newaxis] + 255 * (1 - mask[:, :, np.newaxis])).astype(np.uint8)

        face_cartoon = self.infer(crop_face)
        height, width, _ = face_cartoon.shape
        exp_hei = int(height * 1.05)
        exp_wid = int(width * 1.05)
        h_o = (exp_hei - height) // 2
        w_o = (exp_wid - width) // 2
        face_cartoon = face_cartoon[h_o:(height - h_o), w_o:(width - w_o)]
        face_cartoon = cv2.resize(face_cartoon, (width, height), interpolation=cv2.INTER_LANCZOS4)

        if self.apply_blend is False:
            return cv2.resize(face_cartoon, (self.inp_w, self.inp_h), interpolation=cv2.INTER_LANCZOS4)

        if self.apply_proc_bg and self.custom_bg_func is not None:
            frame = self.custom_bg_func(frame)
        elif self.apply_proc_bg and self.lut_img is not None:
            frame = self.cartoon_bg(frame)
            frame = self.bg_lut(frame)
        else:
            frame = self.cartoon_bg(frame)

        if self.apply_seg:
            bg_cart = frame[box[1]:box[3], box[0]:box[2], :]
            overlay_cartoon_face = (1 - mask[:, :, np.newaxis]) * bg_cart + mask[:, :, np.newaxis] * face_cartoon
            overlay_cartoon_face = overlay_cartoon_face.astype(np.uint8)
        else:
            overlay_cartoon_face = face_cartoon

        if self.apply_lut_to_cartoon_face:
            overlay_cartoon_face = self.cartoon_bg(overlay_cartoon_face)

        frame[box[1]:box[3], box[0]:box[2], :] = overlay_cartoon_face

        self.call_counter += 1
        return frame
