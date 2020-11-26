#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/18 3:03 PM
# @Author  : edvardzeng
# @File    : tips.py
# @Software: PyCharm


def coreml_infer_snippet():
    print("""
    # Use PIL to load and resize the image to expected size
    from PIL import Image
    example_image = Image.open("daisy.jpg").resize((224, 224))
    
    # Make a prediction using Core ML
    out_dict = model.predict({"input_1": example_image})
    
    # Print out top-1 prediction
    print(out_dict["classLabel"])
    """)


def onnx_infer_snippet():
    print("""
    import numpy
    import onnxruntime as rt
    
    sess = rt.InferenceSession("logreg_iris.onnx")
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
    print(pred_onx)    
    """)


def plot_img_snippet():
    print("""
    import matplotlib.pyplot as plt
    plt.figure()
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    
    plt.subplot(1, 3, 3)
    plt.imshow(img3)
    
    plt.show()
    """)


if __name__ == '__main__':
    pass
