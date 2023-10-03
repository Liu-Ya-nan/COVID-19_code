import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import pandas as pd

from train import *
from LearningCurve import *
from predictions import *

# 放训练集/验证集/测试集的NII病人图片的图片文件夹!!!类似下面两个..
data_train = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_train'
data_val = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_val'
data_test = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_test'
# data_train = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/test_D/l348nii_data_train'
# data_val = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/test_D/l348nii_data_val'
# data_test = '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/l348nii_data/l348nii_data_val'

print('Using', torch.cuda.device_count(), 'GPUs')


def main():
    MODE = "train"  # Select "train" or "test", "Resume", "plot"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if MODE == "train":
        ModelType = 'DenseNetModel'
        sample_size = 256
        sample_duration = 348
        LR = 0.001
        MAX_EPOCH = 100
        batch_size = 1
        print('开始训练')

        train(ModelType, data_train, data_val, sample_size, sample_duration, device, LR, MAX_EPOCH, batch_size)  # ModelTrain 在 train.py里面

        PlotLearnignCurve()

    if MODE == "test":
        # val_df = pd.read_csv(data_val)
        # test_df = pd.read_csv(data_test)

        # 当使用torch.load加载模型参数时，会默认加载在第一块GPU0上。
        # 指定GPU映射，将GPU0映射到GPU2（任意一张空闲显卡）
        CheckPointData = torch.load('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/best_acc.pth.tar')
        model = CheckPointData['model']

        make_pred_multilabel(model, data_test, data_val, device)


    if MODE == "Resume":
        ModelType = "Resume"
        CriterionType = 'BCELoss'
        sample_size = 256
        sample_duration = 348
        LR = 0.001
        MAX_EPOCH = 100
        batch_size = 1
        print('开始断点连接')

        model, best_epoch = train(ModelType, data_train, data_val, sample_size, sample_duration, device, LR, MAX_EPOCH, batch_size)

        PlotLearnignCurve()


if __name__ == "__main__":
    main()
