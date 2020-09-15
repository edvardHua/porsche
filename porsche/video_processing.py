#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 5:13 PM
# @Author  : edvardzeng
# @File    : video_proc.py
# @Software: PyCharm


import os
import time
import cv2
import traceback
import numpy as np
from copy import deepcopy
from utils.army_knife import random_string, get_file_name
from utils.img_utils import dynamic_ratio_resize, padding_img


def proc_video(in_path, proc_method, is_test=False, out_path=None, rotate_deg=None, vs_mode=True, specify_fps=None,
               scale_video=True):
    """

    :param in_path: video input path
    :param out_path: video output path
    :param proc_method: func apply to frame, the return must be frame or frame list
    :param rotate_deg: cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
    :param vs_mode: if True, will append result to output result
    :param specify_fps:
    :param scale_video: boolean
    :return:
    """
    os.makedirs("tmp_dir", exist_ok=True)
    cap = cv2.VideoCapture(in_path)

    file_name = get_file_name(in_path)
    if out_path is None:
        fns = os.path.split(in_path)
        out_path = fns[0] + "/res_{}_{}.mp4".format(file_name, random_string(4))
    print("Output path {}".format(out_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    SCALE_SIDE = 520
    scale_flag = False
    if scale_video:
        if width >= height and height > SCALE_SIDE:
            width = int((width / height) * SCALE_SIDE)
            height = SCALE_SIDE
            scale_flag = True
        elif width < height and width > SCALE_SIDE:
            height = int((height / width) * SCALE_SIDE)
            width = SCALE_SIDE
            scale_flag = True

    total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Total frame = %d" % total_frame_count)
    # video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (int(width), int(height)))]

    video_writer = None
    flag = True
    counter = 0
    while flag:
        if is_test and counter > 10:
            break

        st = time.time()
        flag, frame = cap.read()

        if frame is None:
            break

        if scale_flag and scale_video:
            frame = cv2.resize(frame, (width, height))

        if rotate_deg is not None:
            frame = cv2.rotate(frame, rotate_deg)
            height, width, _ = frame.shape

        infer_cost = 0
        try:
            res = proc_method(deepcopy(frame))
            infer_cost = (time.time() - st) * 1000
        except Exception as e:
            cv2.imwrite("tmp_dir/{}_{}.jpg".format(file_name, counter), frame)
            print("Failed to proc {} th frame".format(counter), e)
            print(traceback.format_exc())
            counter += 1
            continue

        counter += 1

        final_res = __concat_proc_result(res, width, height)

        if vs_mode:
            if width >= height:
                write_res = np.vstack([frame, final_res])
            else:
                write_res = np.hstack([frame, final_res])
        else:
            write_res = final_res

        if video_writer is None:
            h, w, _ = write_res.shape
            if specify_fps is not None:
                fps = specify_fps
            video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        video_writer.write(write_res)
        total_cost = (time.time() - st) * 1000
        print("Proc %d/%d" % (counter, total_frame_count),
              "proc method cost %.2f ms, total cost %2.f ms" % (infer_cost, total_cost))

    if video_writer is not None:
        video_writer.release()
    print("Done...")


def __concat_proc_result(res, width, height):
    def nest_proc(img):
        if width >= height:
            tmp_img = dynamic_ratio_resize(img, dest_height=height)
        else:
            tmp_img = dynamic_ratio_resize(img, dest_width=width)
        return padding_img(tmp_img, (width, height))[0]

    if isinstance(res, list):
        all_imgs = [nest_proc(_) for _ in res]
        if width >= height:
            return np.vstack(all_imgs)
        else:
            return np.hstack(all_imgs)
    else:
        return nest_proc(res)


if __name__ == '__main__':
    pass
