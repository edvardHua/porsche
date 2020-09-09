#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 5:13 PM
# @Author  : edvardzeng
# @File    : video_proc.py
# @Software: PyCharm


import os
import cv2
import numpy as np
from utils.army_knife import random_string, get_file_name
from utils.img_utils import dynamic_ratio_resize, padding_img


def proc_video(in_path, proc_method, out_path=None, rotate_deg=None, vs_mode=True, specify_fps=None):
    """

    :param in_path: video input path
    :param out_path: video output path
    :param proc_method: func apply to frame, the return must be frame or frame list
    :param rotate_deg: cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
    :param vs_mode: if True, will append result to output result
    :return:
    """

    cap = cv2.VideoCapture(in_path)

    if out_path is None:
        fns = os.path.split(in_path)
        out_path = fns[0] + "/{}.mp4".format(random_string(10))

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Total frame = %d" % total_frame_count)
    # video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (int(width), int(height)))]

    file_name = get_file_name(in_path)

    video_writer = None
    flag = True
    counter = 0
    while flag:

        flag, frame = cap.read()

        print("Proc %d/%d" % (counter, total_frame_count))

        if frame is None:
            break

        if rotate_deg is not None:
            frame = cv2.rotate(frame, rotate_deg)
        # break

        res = None

        try:
            res = proc_method(frame)
        except:
            cv2.imwrite("tmp_dir/{}_{}.jpg".format(file_name, counter), frame)
            print("Failed to proc {}")
            counter += 1
            continue

        counter += 1

        # if counter > 100:
        #     break

        final_res = __concat_proc_result(res, width, height)

        if vs_mode:
            if width >= height:
                write_res = np.vstack([final_res, frame])
            else:
                write_res = np.hstack([final_res, frame])
        else:
            write_res = final_res

        if video_writer is None:
            h, w, _ = write_res.shape
            if not specify_fps:
                fps = specify_fps
            video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        video_writer.write(write_res)

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
