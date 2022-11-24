# -*- coding: utf-8 -*-
# @Time : 2022/11/24 17:28
# @Author : zihua.zeng
# @File : torch_knife.py

import torch
import numpy as np


def tensor2img(t, mean=None, std=None):
    """
    normally, t.shape = (b, c, h, w) or (c, h, w)
    @return: list of image arr
    """
    ndarr = t.detach().numpy()
    if mean and std:
        ndarr = (ndarr * std + mean) * 255
    else:
        ndarr *= 255.

    if len(ndarr.shape) == 4:
        outs = []
        for _ in range(ndarr.shape[0]):
            cur = ndarr[0]
            cur = np.transpose(cur, (2, 0, 1))
            cur = np.clip(cur, 0, 255)
            outs.append(cur)
        return outs
    elif len(ndarr.shape) == 3:
        ndarr = np.transpose(ndarr, (2, 0, 1))
        ndarr = np.clip(ndarr, 0, 255)
        return [ndarr.astype(np.uint8)]
    else:
        raise RuntimeError("Tensor shape error.")


def imgs2tensor(inp, mean=None, std=None):
    """
    @inp: list of cv2 mat
    """
    temp = []
    for i in inp:
        i = (i.astype(np.float32) / 255.)

        if mean and std:
            i = (i - mean) / std

        i = np.transpose(i, [2, 0, 1])
        i = np.expand_dims(i, axis=0)
        temp.append(torch.from_numpy(i))

    return torch.cat(temp)


if __name__ == '__main__':
    pass
