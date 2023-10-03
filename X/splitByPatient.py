# @Time: 2022/5/8 15:25

# @Author: lyn

# @File: splitData.py

# -*- coding: utf-8 -*-
import csv
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

root = '/home/qwe/data/cuihaihang/lyn/X/X_data'
data_path = '/home/qwe/data/cuihaihang/lyn/X/metadata_X_two.csv'
train_path = '/home/qwe/data/cuihaihang/lyn/X/new_train.csv'
val_path = '/home/qwe/data/cuihaihang/lyn/X/new_val.csv'
test_path = '/home/qwe/data/cuihaihang/lyn/X/new_test.csv'


def split_train_val():
    random.seed(66)  # 保证随机结果可复现18
    # 使用assert可以在出现有异常的代码处直接终止运行。 而不用等到程序执行完毕之后抛出异常。
    # 返回值为假时就会触发异常
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个病人 root:14400
    patient_list = [patient for patient in os.listdir(root) if os.path.isdir(os.path.join(root, patient))]
    print('{} patients in the whole dataset'.format(len(patient_list)))
    print("已合并病人")
    # 8:1:1划分病人数据集
    patient_train, patient_val_test = train_test_split(patient_list, train_size=0.8)
    print('{} patients in train dataset'.format(len(patient_train)))  # 11520
    patient_val, patient_test = train_test_split(patient_val_test, train_size=0.5)
    print('{} patients in val dataset'.format(len(patient_val)))  # 1440
    print('{} patients in test dataset'.format(len(patient_test)))  # 1440
    print("划分了病人")
    data = pd.read_csv(data_path)  # 共23514张图片
    column = list(data.columns)
    print('{} images in all dataset'.format(len(data)))
    print("读取图片")
    # 将图片数据划分为训练集、验证集、测试集
    with open(train_path, 'w', newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(column)
        with open(val_path, 'w', newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow(column)
            with open(test_path, 'w', newline='') as f3:
                writer3 = csv.writer(f3)
                writer3.writerow(column)
                for index in range(len(data)):
                    patient = data.iloc[index]
                    pId = patient['Pid']
                    if pId in patient_train:
                        writer1.writerow(patient)
                    elif pId in patient_val:
                        writer2.writerow(patient)
                    else:
                        writer3.writerow(patient)

    train = pd.read_csv(train_path)
    print('{} images in train dataset'.format(len(train)))  # 19101

    val = pd.read_csv(val_path)
    print('{} images in val dataset'.format(len(val)))  # 2098

    test = pd.read_csv(test_path)
    print('{} images in test dataset'.format(len(test)))  # 2315
    print("分好三个csv文件")


split_train_val()
