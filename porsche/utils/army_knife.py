#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 5:33 PM
# @Author  : edvardzeng
# @File    : army_knife.py
# @Software: PyCharm

import os
import cv2
import string, random


def random_string(str_len=20):
    return ''.join(random.sample(string.ascii_letters + string.digits, str_len))


def get_file_name(path):
    fns = os.path.split(path)
    fn = fns[-1]
    return fn[:fn.rfind(".")]


def get_file_suffix(path):
    fns = os.path.split(path)
    fn = fns[-1]
    return fn[fn.rfind("."):]


def get_porsche_base_path():
    fn = __file__
    ind = fn.rfind("porsche")
    return os.path.join(fn[:ind], "porsche")