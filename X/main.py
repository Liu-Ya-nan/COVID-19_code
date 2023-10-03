import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import torch
from train import *

from LearningCurve import *
from predictions import *
import pandas as pd
from plot import *

path_image = '/home/qwe/data/cuihaihang/lyn/plan4/train/train-plan4/'
# test_path_image = '/home/qwe/data/cuihaihang/lyn/plan1/test-plan1/'
test_path_image = '/home/qwe/data/cuihaihang/lyn/plan5/test/test-plan5/'

train_df_path = "/home/qwe/data/cuihaihang/lyn/plan4/train/train.csv"
val_df_path = "/home/qwe/data/cuihaihang/lyn/plan4/train/val.csv"
# test_df_path = "/home/qwe/data/cuihaihang/lyn/plan1/test.csv"
test_df_path = "/home/qwe/data/cuihaihang/lyn/plan5/test/test.csv"


# diseases = ['No Finding', 'COVID-19', 'Normal']
diseases = ['COVID-19']

# Age = ['0-20', '20-40', '40-60', '60-80', '80-']
Age = ['60-80', '40-60', '20-40', '80-', '0-20']
gender = ['M', 'F']


def main():
    MODE = "test"  # Select "train" or "test", "Resume", "plot"

    # torch.cuda.set_device(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # device = torch.device("cuda")  # cuda 指定使用GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df size", train_df_size)

    test_df = pd.read_csv(test_df_path)
    test_df_size = len(test_df)
    print("Test_df size", test_df_size)

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("Validation_df size:", val_df_size)


    if MODE == "train":
        ModelType = "densenet121"  # select 'ResNet50','densenet121','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.0005
        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device, LR)   # ModelTrain 在 train.py里面

        PlotLearnignCurve()           # PlotLearnignCurve 在 LearningCurve.py里面

    if MODE == "test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        # 当使用torch.load加载模型参数时，会默认加载在第一块GPU0上。
        # 指定GPU映射，将GPU0映射到GPU2（任意一张空闲显卡）
        CheckPointData = torch.load('/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/checkpoint.tar')
        # CheckPointData = torch.load(r'D:\dataset\CheXpert-v1.0-small\results\checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, test_path_image, device)

    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet121','ResNet34', 'densenet169', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.0005

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device, LR)

        PlotLearnignCurve()

    # 计算：
    # Run1_sex: %M男性真实标签阳性占所有人的比例； Gap_M:男性TPR-女性TPR
    if MODE == "plot":
        print('----开始绘图19----')
        # gt = pd.read_csv("./results/True.csv")
        pred = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/bipred.csv")
        factor = [gender, Age]
        factor_str = ['Sex', 'Age']
        for i in range(len(factor)):
            plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_CXP(pred, diseases, factor[i], factor_str[i])
            # plot_sort_14(pred, diseases, factor[i], factor_str[i])
            # distance_max_min(pred, diseases, factor[i], factor_str[i])
            # plot_14(pred, diseases, factor[i], factor_str[i])
    # if MODE == "mean":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [gender, age_decile]
    #     factor_str = ['Sex', 'Age']
    #     for i in range(len(factor)):
    #         mean(pred, diseases, factor[i], factor_str[i])

    # if MODE == "plot_14":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [Age]
    #     factor_str = ['Age']
    #     for i in range(len(factor)):
    #         plot_14(pred, diseases, factor[i], factor_str[i])
    #         plot_Median(pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
