#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 5:15 PM
# @Author  : edvardzeng
# @File    : setup.py.py
# @Software: PyCharm

from setuptools import setup, find_packages

# 需要将那些包导入
# packages = ["video_proc"]

# 导入静态文件
# file_data = [
#     ("smart/static", ["smart/static/icon.svg", "smart/static/config.json"]),
# ]

# 第三方依赖
# requires = [
#     "opencv-python>4.0"
# ]


# 自动读取version信息
# about = {}
# with open(os.path.join(here, 'smart', '__version__.py'), 'r', 'utf-8') as f:
#     exec(f.read(), about)

# 自动读取readme
# with open('README.rst', 'r', 'utf-8') as f:
#     readme = f.read()

setup(
    name="porsche",  # 包名称
    version="0.0.7",  # 包版本
    description="Collection of mess function...",  # 包详细描述
    long_description="Collection of mess function...",  # 长描述，通常是readme，打包到PiPy需要
    author="edvardzeng",  # 作者名称
    author_email="edvard_hua@live.com",  # 作者邮箱
    url="http://www.xxx.com",  # 项目官网
    packages=find_packages(),  # 项目需要的包
    include_package_data=True,  # 是否需要导入静态数据文件
    # python_requires=">=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3*",  # Python版本依赖
    # install_requires=requires,  # 第三方库依赖
)
