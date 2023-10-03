# @Time: 2022/5/8 15:25

# @Author: lyn

# @File: split.py

# -*- coding: utf-8 -*-
import csv
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

root = '/home/qwe/data/cuihaihang/lyn/X/X_data/'
data_path = '/home/qwe/data/cuihaihang/lyn/X/metadata_X_two.csv'
train_path = '/home/qwe/data/cuihaihang/lyn/X/new_train.csv'
val_path = '/home/qwe/data/cuihaihang/lyn/X/new_val.csv'
test_path = '/home/qwe/data/cuihaihang/lyn/X/new_test.csv'

def split_train_val():
    random.seed(66)  # 保证随机结果可复现18
    # 使用assert可以在出现有异常的代码处直接终止运行。 而不用等到程序执行完毕之后抛出异常。
    # 返回值为假时就会触发异常
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    data = pd.read_csv(data_path)
    # print(data)
    patient_train, patient_val_test = train_test_split(data, train_size=0.8)
    patient_val, patient_test = train_test_split(patient_val_test, train_size=0.5)

    patient_train.to_csv(train_path)
    patient_val.to_csv(val_path)
    patient_test.to_csv(test_path)
    print('{} patients in train and val dataset'.format(len(patient_train)))  # 30931 ，用来十折交叉验证
    print('{} patients in val and val dataset'.format(len(patient_val)))  # 3866
    print('{} patients in test dataset'.format(len(patient_test)))  # 3867
    print("已划分了测试图片数量")


split_train_val()

