---
title: "Dicom文件处理"
author: "wuyangzz"
tags: [""]
categories: [""]
date: 2021-10-09T10:17:39+08:00
---

# 使用阈值去掉杂乱的信息
```python
import cv2
import os
import pydicom
import numpy as np
# dicom图像输入路径
inputdir = '/workspace/20210910/Bmodel/inputs'
# dicom图像输出路径
outdir = '/workspace/20210910/Bmodel/max_contours_images/'

# 获取文件夹下文件列表
files_name_list=os.listdir(inputdir)
count=0
# 遍历所有文件

def findMaxcontours(data):
    _, binaryzation = cv2.threshold(data, 2, 255,
                                    cv2.THRESH_BINARY_INV)
    binaryzation = np.uint8(binaryzation)
    # 找到所有的轮廓
    contours, _ = cv2.findContours(
        binaryzation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    # 找到最大的轮廓 实际应该为第二大轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    # max_idx = np.argmax(np.array(area))
    max_idx = np.argsort(np.array(area))[-2]
    # cv2.fillContexPoly(mask[i], contours[max_idx], 0)
    # 填充最大的轮廓
    mask = cv2.drawContours(np.zeros(data.shape), contours,
                            max_idx, 1, cv2.FILLED)

    image1 = np.multiply(mask, data)
    image1 = cv2.cvtColor(np.uint8(image1), cv2.COLOR_GRAY2BGR)
    return image1

for file_name in files_name_list:
    path=os.path.join(inputdir,file_name)
    ds=pydicom.read_file(path)
    # 获取该文件的帧数
    num_frame=ds.pixel_array.shape[0]
    # 逐帧保存为PNG无损图像
    for i in range(num_frame):
        if i == 0:
            image1 = findMaxcontours(ds.pixel_array[i, :, :])
        else:
            image2 = findMaxcontours(ds.pixel_array[i, :, :])
            cv2.imwrite(os.path.join(outdir, str(
                count)+"_1.png"), image1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(os.path.join(outdir, str(
                count)+"_2.png"), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            image1=image2
            count+=1

```

dicom处理为tf train数据格式
```python
import cv2
import os
import pydicom
import numpy as np
import tensorflow as tf

import conversion_utils




# 遍历所有文件


def write_data_example(record_writer, image1, image2):
    """Write data example to disk."""
    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    # assert image1.shape[2] == image2.shape[2]

    feature = {
        'height': conversion_utils.int64_feature(image1.shape[0]),
        'width': conversion_utils.int64_feature(image1.shape[1]),
    }
    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=feature),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'images':
                    tf.train.FeatureList(feature=[
                        conversion_utils.bytes_feature(
                            image1.astype('uint8').tobytes()),
                        conversion_utils.bytes_feature(
                            image2.astype('uint8').tobytes())
                    ]),
            }))
    record_writer.write(example.SerializeToString())


def findMaxcontours(data):
    _, binaryzation = cv2.threshold(data, 2, 255,
                                    cv2.THRESH_BINARY_INV)
    binaryzation = np.uint8(binaryzation)
    # 找到所有的轮廓
    contours, _ = cv2.findContours(
        binaryzation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    # 找到最大的轮廓 实际应该为第二大轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    # max_idx = np.argmax(np.array(area))
    max_idx = np.argsort(np.array(area))[-2]
    # cv2.fillContexPoly(mask[i], contours[max_idx], 0)
    # 填充最大的轮廓
    mask = cv2.drawContours(np.zeros(data.shape), contours,
                            max_idx, 1, cv2.FILLED)

    image1 = np.multiply(mask, data)
    image1 = cv2.cvtColor(np.uint8(image1), cv2.COLOR_GRAY2BGR)
    return image1





if __name__ == '__main__':
    # dicom图像输入路径
    inputdir = '/workspace/20210910/Bmodel/inputs'
    # dicom图像输出路径
    outdir_tfcard = '/workspace/20210910/Bmodel/max_contours_tfcard_size512*512_间隔1帧_RGB'
    outdir_image = '/workspace/20210910/Bmodel/max_contours_images_size512*512_间隔1帧_RGB'
    if not tf.io.gfile.exists(outdir_tfcard):
        print('Making new tfcard directory', outdir_tfcard)
        tf.io.gfile.makedirs(outdir_tfcard)
    if not tf.io.gfile.exists(outdir_image):
        print('Making new image directory', outdir_image)
        tf.io.gfile.makedirs(outdir_image)
    filename = os.path.join(outdir_tfcard, 'fdicom@1')
    files_name_list = os.listdir(inputdir)
    count = 0
    with tf.io.TFRecordWriter(filename) as record_writer:
        for file_name in files_name_list:
            # 获取文件夹下文件列表
            path = os.path.join(inputdir, file_name)
            ds = pydicom.read_file(path)
            # 获取该文件的帧数
            num_frame = ds.pixel_array.shape[0]
            print('Read a new dicom file: ' + file_name +
                                      "      frame: %d", num_frame)
            ds1 = ds.pixel_array[:-1]
            ds2 = ds.pixel_array[1:]
            # 逐帧保存为PNG无损图像
            for i in range(ds1.shape[0]):
                image1 = findMaxcontours(ds1[i, :, :])
                image2 = findMaxcontours(ds2[i, :, :])
                image1=image1[60:572, 114:626, :]
                image2=image2[60:572, 114:626, :]
                cv2.imwrite(os.path.join(outdir_image, str(
                    count)+"_1.png"), image1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(os.path.join(outdir_image, str(
                    count)+"_2.png"), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                write_data_example(record_writer, image1, image2)
                count += 1

            print('Read a new frame: %d', count)

```
20211012修改版本 
```python
import cv2
import os
import pydicom
import numpy as np
import tensorflow as tf

import conversion_utils
import tensorflow as tf
from tensorflow import keras

# 遍历所有文件


def write_data_example(record_writer, image1, image2):
    """Write data example to disk."""
    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    # assert image1.shape[2] == image2.shape[2]

    feature = {
        'height': conversion_utils.int64_feature(image1.shape[0]),
        'width': conversion_utils.int64_feature(image1.shape[1]),
    }
    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=feature),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'images':
                    tf.train.FeatureList(feature=[
                        conversion_utils.bytes_feature(
                            image1.astype('uint8').tobytes()),
                        conversion_utils.bytes_feature(
                            image2.astype('uint8').tobytes())
                    ]),
            }))
    record_writer.write(example.SerializeToString())


def findMaxcontours_mask(data):
    _, binaryzation = cv2.threshold(data*255, 2, 255,
                                    cv2.THRESH_BINARY_INV)
    binaryzation = np.uint8(binaryzation)
    # 找到所有的轮廓
    contours, _ = cv2.findContours(
        binaryzation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    # 找到最大的轮廓 实际应该为第二大轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    # max_idx = np.argmax(np.array(area))
    max_idx = np.argsort(np.array(area))[-2]
    # cv2.fillContexPoly(mask[i], contours[max_idx], 0)
    # 填充最大的轮廓
    mask = cv2.drawContours(np.zeros(data.shape), contours,
                            max_idx, 1, cv2.FILLED)

    return mask


def findMaxcontours_raw(data):
    _, binaryzation = cv2.threshold(data, 2, 255,
                                    cv2.THRESH_BINARY_INV)
    binaryzation = np.uint8(binaryzation)
    # 找到所有的轮廓
    contours, _ = cv2.findContours(
        binaryzation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    # 找到最大的轮廓 实际应该为第二大轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    # max_idx = np.argsort(np.array(area))[-2]
    # cv2.fillContexPoly(mask[i], contours[max_idx], 0)
    # 填充最大的轮廓
    mask = cv2.drawContours(np.ones(data.shape), contours,
                            max_idx, 0, cv2.FILLED)

    image = np.multiply(mask, data)
    return image

if __name__ == '__main__':
    # dicom图像输入路径
    inputdir = '/workspace/20210910/Bmodel/inputs'
    # dicom图像输出路径
    # outdir_tfcard_unet = '/workspace/20210910/datasets/512_512_tf_UNET'
    # outdir_tfcard_raw = '/workspace/20210910/datasets/512_512_tf_RAW'
    # outdir_image_unet = '/workspace/20210910/datasets/512_512_image_UNET'
    # outdir_image_raw = '/workspace/20210910/datasets/512_512_image_RAW'
    outdir_tfcard_unet = '/workspace/20210910/datasets/480_384_tf_UNET'
    outdir_tfcard_raw = '/workspace/20210910/datasets/480_384_tf_RAW'
    outdir_image_unet = '/workspace/20210910/datasets/480_384_image_UNET'
    outdir_image_raw = '/workspace/20210910/datasets/480_384_image_RAW'
    model_path = '/workspace/my-unet3plus/model/图像大小320使用crop进行训练/model'
    model_size = 320
    # 检查输出文件夹
    if not tf.io.gfile.exists(outdir_tfcard_unet):
        print('Making new tfcard directory', outdir_tfcard_unet)
        tf.io.gfile.makedirs(outdir_tfcard_unet)
    if not tf.io.gfile.exists(outdir_tfcard_raw):
        print('Making new image directory', outdir_tfcard_raw)
        tf.io.gfile.makedirs(outdir_tfcard_raw)
    if not tf.io.gfile.exists(outdir_image_unet):
        print('Making new tfcard directory', outdir_image_unet)
        tf.io.gfile.makedirs(outdir_image_unet)
    if not tf.io.gfile.exists(outdir_image_raw):
        print('Making new image directory', outdir_image_raw)
        tf.io.gfile.makedirs(outdir_image_raw)
    
    tfcard_unet_dir = os.path.join(outdir_tfcard_unet, 'tfcard_unet@1')
    tfcard_raw_dir = os.path.join(outdir_tfcard_raw, 'tfcard_raw@1')
    dicom_files_list = os.listdir(inputdir)
    count = 0
    '''导入模型'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    unet3plus = keras.models.load_model(model_path, compile=False)
    with tf.io.TFRecordWriter(tfcard_unet_dir) as record_writer_unet:
        with tf.io.TFRecordWriter(tfcard_raw_dir) as record_writer_raw:
            for dicom_file in dicom_files_list:
                # 获取文件夹下文件列表
                dicom_file_dir = os.path.join(inputdir, dicom_file)
                ds_dicom = pydicom.read_file(dicom_file_dir)
                # 获取该文件的帧数
                num_frame = ds_dicom.pixel_array.shape[0]
                print('读取dicom文件: ' + dicom_file +
                    "      数量为: %d", num_frame)
                # 图像大小 51 2* 512
                # images_read = ds_dicom.pixel_array[:, 60:572, 114:626]
                # 图像大小为480*384
                images_read = ds_dicom.pixel_array[:, 68:548, 238:622]
                # 整理图像
                _, h, w = images_read.shape
                for i in range(images_read.shape[0]):
                    if i == 0:
                        images = findMaxcontours_raw(images_read[i, :, :]).reshape(
                            1, h, w)
                    else:
                        images = np.append(images, findMaxcontours_raw(
                            images_read[i, :, :]).reshape(1, h, w), axis=0)
                # resize 图像好作为模型输入
                for i in range(images.shape[0]):
                    if i==0:
                        resize_images = cv2.resize(
                            images[i, :, :], (model_size, model_size), 
                            interpolation=cv2.INTER_AREA).reshape(1, model_size, model_size)
                    else:
                        resize_images = np.append(resize_images, cv2.resize(
                            images[i, :, :], (model_size, model_size),
                            interpolation=cv2.INTER_AREA).reshape(1, model_size, model_size), axis=0)
                # 预测图像
                resize_mask = unet3plus.predict([(resize_images / 255).reshape(
                    images.shape[0], model_size, model_size, 1)],
                    batch_size=16)[-1].reshape(
                    images.shape[0], model_size,
                    model_size)
                # 整理预测后的图像并且从新resize为原大小
                resize_mask = np.uint8(resize_mask>0.3)
                _, h, w = images.shape
                for i in range(images.shape[0]):
                    if i == 0:
                        masks = findMaxcontours_mask(cv2.resize(
                            resize_mask[i, :, :], (w, h),
                            interpolation=cv2.INTER_NEAREST)).reshape(1, h, w)
                    else:
                        masks = np.append(masks, findMaxcontours_mask(cv2.resize(
                            resize_mask[i, :, :], (w, h),
                            interpolation=cv2.INTER_NEAREST)).reshape(1, h, w), axis=0)
                # 划分图像
                images_1, images_2 = images[:-1], images[1:]
                masks_1, masks_2 = masks[:-1],masks[1:]
                # 逐帧保存为PNG无损图像
                for i in range(images_1.shape[0]):
                    image1, image2 = np.uint8(images_1[i,:, :]), np.uint8(images_2[i, :, :])
                    mask1, mask2 = masks_1[i, :, :], masks_2[i, :, :]
                    # 膨胀运算
                    kernel_2 = np.ones((4, 4), dtype=np.uint8)
                    mask1, mask2 = np.uint8(cv2.dilate(
                        mask1, kernel_2, 1)), np.uint8(cv2.dilate(mask2, kernel_2, 1))
                    # 进行叠加
                    mask_1_2 = np.uint8(np.add(mask1, mask2) > 0.9)
                    mask1_sum, mask2_sum, mask_1_2_sum = mask1.sum(), mask2.sum(), mask_1_2.sum()

                    if mask1_sum < 13000 or mask2_sum < 13000 or mask_1_2_sum<13000:
                        continue
                    else:
                        image1_unet = np.multiply(
                            image1, mask_1_2)
                        image2_unet = np.multiply(
                            image2, mask_1_2)
                        # 保存原图
                        cv2.imwrite(os.path.join(outdir_image_raw, str(
                            count)+"_1.png"), image1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        cv2.imwrite(os.path.join(outdir_image_raw, str(
                            count)+"_2.png"), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        image1=cv2.cvtColor(
                            np.uint8(image1), cv2.COLOR_GRAY2BGR)
                        image2 = cv2.cvtColor(
                            np.uint8(image2), cv2.COLOR_GRAY2BGR)
                        write_data_example(record_writer_raw, image1, image2)
                        # 保存unet图
                        cv2.imwrite(os.path.join(outdir_image_unet, str(
                            count)+"_1.png"), image1_unet, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        cv2.imwrite(os.path.join(outdir_image_unet, str(
                            count)+"_2.png"), image2_unet, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        image1_unet = cv2.cvtColor(
                            np.uint8(image1_unet), cv2.COLOR_GRAY2BGR)
                        image2_unet = cv2.cvtColor(
                            np.uint8(image2_unet), cv2.COLOR_GRAY2BGR)
                        write_data_example(
                            record_writer_unet, image1_unet, image2_unet)
                        count += 1

                print('总对数为: %d', count)

```


```python
import numpy as np
from glob import glob
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max
from tensorflow.python.ops.math_ops import to_int32
from keras_unet_collection import models, base, utils
from keras_unet_collection import losses
import os
import cv2
import sys
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def plot_un(resize_img, resize_mask, plot_dir, name):
    # 保存resize后的数据
    fig_height, fig_width=6, 6
    plt.figure(figsize=(fig_width * 3, fig_height * 1))
    plt.subplot(1, 3, 1)
    plt.title('RESIZE_IMG', fontsize=10)
    plt.axis('off')
    plt.imshow(resize_img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('MASK_IMG', fontsize=10)
    plt.imshow(resize_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    resize_img_copy=resize_img.copy()
    _, resize_mask_1=cv2.threshold(resize_mask, 125, 255,
                                    cv2.THRESH_BINARY_INV)
    # gray = cv2.cvtColor(resize_mask, cv2.COLOR_BGR2GRAY)
    resize_mask_1=resize_mask_1.astype(np.uint8)
    contours, _=cv2.findContours(
        resize_mask_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area=[]
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    # max_idx = np.argmax(np.array(area))
    max_idx = np.argsort(np.array(area))[-2]

    cv2.drawContours(resize_img_copy, contours,
                        max_idx, 255, thickness=3)
    plt.title('IMG', fontsize=10)
    plt.imshow(resize_img_copy, cmap='gray')
    plt.savefig(os.path.join(plot_dir, name),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()

def plot_image(raw_img,resize_img,resize_mask,plot_dir,id):
    # 保存原始数据
    if resize_mask.sum()<1300000:
        print('\n'+os.path.join(plot_dir, str(id)))
        return
    if not os.path.exists(os.path.join(plot_dir)):
        os.makedirs(os.path.join(plot_dir))
    cv2.imwrite(os.path.join(plot_dir, str(id)+"_原图.png"), raw_img)
    cv2.imwrite(os.path.join(plot_dir, str(
        id)+"_resize.png"), resize_img)
    cv2.imwrite(os.path.join(plot_dir, str(id)+"_mask.png"),
                resize_mask)
    mask_resize_int = np.uint8(cv2.resize(
        resize_mask, (raw_img.shape[1], raw_img.shape[0])) > 125)
    cv2.imwrite(os.path.join(plot_dir, str(id)+"_unet.png"),
                np.multiply(raw_img, mask_resize_int))
    # cv2.imwrite()
    try:
        plot_un(resize_img, resize_mask, plot_dir, str(id)+"_resize_predict.png")
        plot_un(raw_img, cv2.resize(resize_mask, (raw_img.shape[1], raw_img.shape[0])),
            plot_dir, str(id)+"_predict.png")
    except:
        print('\n'+plot_dir+str(id)+"_predict.png")

def predict_dataset(dataset_folder):
    images = os.listdir(dataset_folder)
    # images = [_ for _ in os.listdir(dataset_folder) if _.endswith("_1.png")]
    images = [os.path.join(dataset_folder, i) for i in images]
    images = sorted(images,
                    key=lambda x: int(os.path.basename(x).split('.')[0]))
    for image in images:
        image = cv2.imread(image, 0)
        yield image


def predict_main(dataset_folder, model_folder, model_image_size, plot_folder):
    model = keras.models.load_model(model_folder, compile=False)
    print("Start Predict "+plot_folder)
    for i, image in enumerate(predict_dataset(dataset_folder)):
        sys.stdout.write(f'{i},')
        sys.stdout.flush()
        image_resize = cv2.resize(image, (model_image_size, model_image_size))
        infer_mask = model.predict(
            (image_resize / 255).reshape(1, image_resize.shape[0],
                                         image_resize.shape[0], 1))[-1]
        # cv2.imwrite(os.path.join(plot_folder, str(i)+"_原图.png"), image)
        # cv2.imwrite(os.path.join(plot_folder, str(
        #     i)+"_resize.png"), image_resize)
        # cv2.imwrite(os.path.join(plot_folder, str(i)+"_mask.png"),
        #             infer_mask[0, :, :, 0]*255)
        plot_image(image, image_resize,
                   infer_mask[0, :, :, 0]*255, plot_folder, i)
        
    print("End Predict")


if __name__ == '__main__':
    dataset_folder = "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0010"
    model_folder = "/workspace/my-unet3plus/model/图像大小320使用crop进行训练/model"
    plot_folder = "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0010_UNET"
    model_image_size = 320
    # predict_main(dataset_folder, model_folder, model_image_size, plot_folder)
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0009",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0009_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0010",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0010_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0011",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0011_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0112",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0112_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0113",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0113_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0114",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0114_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0115",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0115_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0158",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0158_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0159",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0159_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0160",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0160_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0161",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0161_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0163",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0163_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0165",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0165_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0166",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0166_UNET")
    predict_main("/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0167",
                 model_folder, model_image_size, "/workspace/20210910/Bmodel/直接crop为512*512灰度图像/IM_0167_UNET")
```
