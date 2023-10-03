import os
import random
from sklearn.model_selection import train_test_split
import shutil

filePath = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/CT_nii_data_348_label'# 用于获取文件名称列表
data_train = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_train'  # 目标文件夹
data_val = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_val'
data_test = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_test'

def split_train_val():
    random.seed(66)  # 保证随机结果可复现18
    # 使用assert可以在出现有异常的代码处直接终止运行。 而不用等到
    # 程序执行完毕之后抛出异常。
    # 返回值为假时就会触发异常
    assert os.path.exists(filePath), "dataset root: {} does not exist.".format(filePath)
    # 遍历文件夹，一个文件夹对应一个病人 root:719
    file_list = os.listdir(filePath)   # file_list就是patient_list
    print('{} patients in the whole dataset'.format(len(file_list)))
    print("已合并病人")
    # 8:1:1划分病人数据集
    patient_train, patient_val_test = train_test_split(file_list, train_size=0.8)
    print('{} patients in train dataset'.format(len(patient_train)))  # 2208
    patient_val, patient_test = train_test_split(patient_val_test, train_size=0.5)
    print('{} patients in val dataset'.format(len(patient_val)))  # 276
    print('{} patients in test dataset'.format(len(patient_test)))  # 276
    print("划分了病人")

    for i in range(len(patient_train)):
        if patient_train[i] in file_list:
            # print('源文件："' + patient_train[i] + '",绝对路径：' + filePath + '/' + patient_train[i])
            # print('目标文件夹：' + new_path + '/' + patient_train[i])
            shutil.copy(filePath + '/' + patient_train[i], data_train + '/' + patient_train[i])
    for i in range(len(patient_val)):
        if patient_val[i] in file_list:
            # print('源文件："' + patient_train[i] + '",绝对路径：' + filePath + '/' + patient_train[i])
            # print('目标文件夹：' + new_path + '/' + patient_train[i])
            shutil.copy(filePath + '/' + patient_val[i], data_val + '/' + patient_val[i])
    for i in range(len(patient_test)):
        if patient_test[i] in file_list:
            # print('源文件："' + patient_train[i] + '",绝对路径：' + filePath + '/' + patient_train[i])
            # print('目标文件夹：' + new_path + '/' + patient_train[i])
            shutil.copy(filePath + '/' + patient_test[i], data_test + '/' + patient_test[i])
    print("复制完成")

split_train_val()



# # 数一数文件数够了么
# import os
# path = '/home/fengxiufang/cuihaihang/lyn/data_3-1/CT2/l348nii_data/l348nii_data_val'      # 输入文件夹地址
# files = os.listdir(path)   # 读入文件夹
# print(len(files))      # 统计文件夹中的文件个数,files是list
