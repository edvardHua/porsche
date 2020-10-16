#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 5:13 PM
# @Author  : edvardzeng
# @File    : video_proc.py
# @Software: PyCharm


import os
import cv2
import time
import math
import traceback
import numpy as np
from copy import deepcopy
from porsche.utils.army_knife import random_string, get_file_name
from porsche.utils.img_utils import dynamic_ratio_resize, padding_img


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


def get_cv2_video_writer(fps=24, size=(480, 640), out_path="out.mp4"):
    """

    :param fps:
    :param size:
    :param out_path:
    :return: writer
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    return video_writer


def sample_video(in_path, out_path=None, sample_rate=2, proc_method=None, skip_head=None, skip_tail=None):
    """
    Sample video by average duration
    :param in_path:
    :param out_path:
    :param sample_rate:
    :param proc_method:
    :return:
    """
    cap = cv2.VideoCapture(in_path)
    filename = get_file_name(in_path)
    if out_path is None:
        in_folder = in_path.split("/")[-2]
        out_folder = in_folder + "_out"

        in_prefix = os.path.split(in_path)[0]
        out_folder_path = in_prefix.replace(in_folder, out_folder)
        os.makedirs(out_folder_path, exist_ok=True)
        out_path = os.path.join(out_folder_path, filename)
        os.makedirs(out_path, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    mod_val = int(math.ceil(fps / sample_rate))
    total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    total_sample_frame = 0
    flag = True
    counter = 0
    while flag:

        flag, frame = cap.read()

        if frame is None:
            break

        if proc_method is not None:
            frame = proc_method(frame)

        if skip_head is not None and skip_head > counter:
            counter += 1
            continue

        if counter % mod_val == 0:
            cv2.imwrite(os.path.join(out_path, "%s_%d.jpg" % (random_string(8), counter)), frame)
            total_sample_frame += 1

        counter += 1

        if skip_tail is not None and (counter == (total_frame_count - skip_tail)):
            break

    cap.release()
    print("Done, get %d imgs from %s into %s" % (total_sample_frame, filename, out_path))


def extract_video_frame(in_path, ind=0):
    """

    :param in_path:
    :param ind: 0 mean first, -1 mean latest
    :return:
    """
    cap = cv2.VideoCapture(in_path)
    video_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    flag = True
    counter = 0
    while flag:
        flag, frame = cap.read()

        if ind == -1 and counter == (video_count - 1):
            cap.release()
            return frame

        if frame is None:
            cap.release()
            raise Exception("The request ind is out of range")

        if counter == ind:
            cap.release()
            return frame

        counter += 1

    if ind == -1:
        return frame


def __concat_proc_result(res, width, height):
    def nest_proc(img, pad=True):
        if width >= height:
            tmp_img = dynamic_ratio_resize(img, dest_height=height)
        else:
            tmp_img = dynamic_ratio_resize(img, dest_width=width)
        return padding_img(tmp_img, (width, height))[0] if pad else tmp_img

    if isinstance(res, list):
        all_imgs = [nest_proc(_) for _ in res]
        if width >= height:
            return np.vstack(all_imgs)
        else:
            return np.hstack(all_imgs)
    else:
        return nest_proc(res, False)


if __name__ == '__main__':
    pass
