from dataset import Covid
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def make_pred_multilabel(model, test_df, val_df, path_image, test_path_image, device):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model
    Args:

        model: densenet-121 from torchvision previously fine tuned to training data
        test_df : dataframe csv file
        PATH_TO_IMAGES:
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    BATCH_SIZE = 48
    workers = 12

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_test = Covid(test_df, path_image=test_path_image, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)

    dataset_val = Covid(val_df, path_image=path_image, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    val_loader = torch.utils.data.DataLoader(dataset_val, BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)

    size = len(test_df)
    print("Test _df size :", size)
    size = len(val_df)
    print("val_df size :", size)

    # criterion = nn.BCELoss().to(device)
    model = model.to(device)

    # to find this threshold阈值, first we get the precision and recall withoit this, from there we calculate f1 score,
    # using f1score, we found this theresold which has best precsision and recall.
    # Then this threshold activation are used to calculate our binary output.

    PRED_LABEL = ['COVID-19']

    for mode in ["Threshold", "test"]:
        # create empty dfs
        pred_df = pd.DataFrame(columns=["Path"])
        bi_pred_df = pd.DataFrame(columns=["Path"])
        true_df = pd.DataFrame(columns=["Path"])

        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []            
        
        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "acc"])
            
            Eval = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/Thereshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "COVID-19"].index[0]]]

        # 将预测结果和真实结果都存进df
        for i, data in enumerate(loader):
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()

            batch_size = true_labels.shape

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                probs = outputs.cpu().data.numpy()

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}      # 预测的label
                bi_thisrow = {}   # 测试集上预测值是否>=阈值
                truerow = {}      # 真实的label

                truerow["Path"] = item[j]
                thisrow["Path"] = item[j]
                if mode == "test":
                    bi_thisrow["Path"] = item[j]

                # iterate over each entry in prediction vector; each corresponds to
                # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    # if mode == "test":
                    #     # True/False
                    #     bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                    if mode == "test":
                        # True/False
                        # bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]
                        if probs[j, k] >= thrs[k]:
                            bi_thisrow["bi_" + PRED_LABEL[k]] = '1'
                        else:
                            bi_thisrow["bi_" + PRED_LABEL[k]] = '0'

                # pred_df = pred_df.append(thisrow, ignore_index=True)
                # true_df = true_df.append(truerow, ignore_index=True)
                pred_df = pd.concat([pred_df, pd.DataFrame(thisrow, index = [0])], axis=0, ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame(truerow, index = [0])], axis=0, ignore_index=True)
                if mode == "test":
                    # bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)
                    bi_pred_df = pd.concat([bi_pred_df, pd.DataFrame(bi_thisrow, index = [0])], axis=0, ignore_index=True)
           
            if (i % 200 == 0):
                print('进行到第多少个病人', str(i * BATCH_SIZE))

        for column in true_df:
            if column not in PRED_LABEL:
                continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            
            thisrow = {}
            thisrow['label'] = column
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]
                thisrow['auc'] = np.nan
                thisrow['acc'] = np.nan
                acc_sum = 0
                acc_sum += sum((bi_pred.values.astype(int) - actual.values.astype(int)) == 0)
                print('acc_sum有值吗？？？', acc_sum)

            else:
                thisrow['bestthr'] = np.nan

            try:

                if mode == "test":
                    # 以FPR=FP/(FP+TN)为X轴，以TPR=TP/(TP+FN)为Y轴，求围成的面积,越大越好
                    # thisrow['auc'] = sklm.roc_auc_score(
                    #     actual.values.astype(int), pred.values)
                    thisrow['auc'] = 1

                    # thisrow['auprc'] = sklm.average_preicision_score(
                    #     actual.values.astype(int), pred.values)

                    accuracy = acc_sum * 100 / len(dataset_test)

                    thisrow['acc'] = accuracy

                else:
                    # precision, recall, thresholds
                    print(actual.values)
                    print(len(actual.values))
                    print(pred.values)
                    print(len(pred.values))
                    p, r, t = sklm.precision_recall_curve(actual.values.astype(int), pred.values)
                    print(p)
                    print(r)
                    # Choose the best threshold based on the highest F1 measure
                    # F1 = 2*(p*r)/(r+p)
                    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                    print(f1)
                    bestthr = t[np.where(f1 == max(f1))]
                    print('bestthr on val', bestthr)

                    thrs.append(bestthr)
                    thisrow['bestthr'] = bestthr[0]    # 有问题！

            except BaseException:
                print("can't calculate auc for " + str(column))

            # 验证集上的label 对应的thrshold
            if mode == "Threshold":
                Eval_df = pd.concat([Eval_df, pd.DataFrame(thisrow, index=[0])], axis=0, ignore_index=True)
                # Eval_df = Eval_df.append(thisrow, ignore_index=True)

            if mode == "test":
                TestEval_df = pd.concat([TestEval_df, pd.DataFrame(thisrow, index=[0])], axis=0, ignore_index=True)
                # TestEval_df = TestEval_df.append(thisrow, ignore_index=True)


        # pred_df.to_csv("results/preds.csv", index=False)
        # true_df.to_csv("results/True.csv", index=False)

        if mode == "Threshold":
            pred_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/valPreds.csv", index=False)
            true_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/valTrue.csv", index=False)
            Eval_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/Thereshold.csv", index=False)

        if mode == "test":
            pred_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/testPreds.csv", index=False)
            true_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/testTrue.csv", index=False)
            TestEval_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/TestEval.csv", index=False)
            bi_pred_df.to_csv("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/bipred.csv", index=False)

    # print("AUC ave:", TestEval_df['auc'].sum() / 2.0)

    print("done")

    return pred_df, Eval_df, bi_pred_df, TestEval_df  # , bi_pred_df , Eval_bi_df

