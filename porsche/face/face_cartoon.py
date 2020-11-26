#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 7:26 PM
# @Author  : edvardzeng
# @File    : face_cartoon.py
# @Software: PyCharm


import os
import cv2
import onnxruntime
import numpy as np

from porsche.face.face_detector import FaceDetector
from porsche.face.segmentation import Segmentation
from porsche.utils.army_knife import get_porsche_base_path


class FaceCartoon:
    MODEL_PATH = "models/student_gan_0902.onnx"
    BATCH_PATH = "models/student_gan_0902_dynamic_axes.onnx"

    def __init__(self, bg_lut_file=None, custom_bg_func=None, model_path=None):
        self.fd = FaceDetector(need_expand=True)
        self.sehead = Segmentation(type="head")
        self.seskin = Segmentation(type="skin")
        if model_path is not None:
            self.cartoon = onnxruntime.InferenceSession(model_path)
        else:
            self.cartoon = onnxruntime.InferenceSession(os.path.join(get_porsche_base_path(), self.MODEL_PATH))

        _, _1, self.inp_h, self.inp_w = self.cartoon._sess.inputs_meta[0].shape

        self.custom_bg_func = custom_bg_func
        if bg_lut_file is not None:
            self.lut_img = cv2.imread(bg_lut_file)
        else:
            self.lut_img = None

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
            ret_frame.append(cv2.resize(cartoon_face, (ori_w, ori_h)))
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
        return cv2.resize(cartoon_face, (ori_w, ori_h))

    def input_preprocess(self, img):
        img = cv2.resize(img, (self.inp_w, self.inp_h))
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

    def __call__(self, frame, seg=True, blend=True, proc_bg=True, detect_face=True):
        frame = frame.copy()
        if detect_face:
            bboxes, _ = self.fd(frame)
            if len(bboxes) != 0:
                box = bboxes[0]
                crop_face = frame[box[1]:box[3], box[0]:box[2], :]
        else:
            crop_face = frame

        if seg:
            # skin include neck
            # mask_head = self.sehead(crop_face)
            # mask_skin = self.seskin(crop_face)
            # mask = (mask_head + mask_skin) / 2.0

            # only head
            mask = self.sehead(crop_face)
            mask[mask >= 0.3] = 1.0
            mask = cv2.blur(mask, (3, 3))
            crop_face = (crop_face * mask[:, :, np.newaxis] + 255 * (1 - mask[:, :, np.newaxis])).astype(np.uint8)

        face_cartoon = self.infer(crop_face)
        height, width, _ = face_cartoon.shape
        exp_hei = int(height * 1.05)
        exp_wid = int(width * 1.05)
        h_o = (exp_hei - height) // 2
        w_o = (exp_wid - width) // 2
        face_cartoon = face_cartoon[h_o:(height - h_o), w_o:(width - w_o)]
        face_cartoon = cv2.resize(face_cartoon, (width, height))

        if blend is False:
            return face_cartoon

        if proc_bg and self.custom_bg_func is not None:
            frame = self.custom_bg_func(frame)
        elif proc_bg and self.lut_img is not None:
            frame = self.cartoon_bg(frame)
            frame = self.bg_lut(frame)
        else:
            frame = self.cartoon_bg(frame)

        bg_cart = frame[box[1]:box[3], box[0]:box[2], :]
        overlay_cartoon_face = (1 - mask[:, :, np.newaxis]) * bg_cart + mask[:, :, np.newaxis] * face_cartoon
        overlay_cartoon_face = overlay_cartoon_face.astype(np.uint8)
        frame[box[1]:box[3], box[0]:box[2], :] = overlay_cartoon_face

        return frame
