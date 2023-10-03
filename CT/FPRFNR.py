import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 统计男女、年龄组之间的percentage和FPR、FNR
def FPFN_NF_Covid(TrueWithMeta_df, df, diseases, category, category_name):
    # return FPR and FNR per subgroup and the unber of patients with 0 No-finding in test set.
    # 这是最后传入的参数，可以参考↓
    # FPFN_NF_Covid(TrueWithMeta, pred, diseases, Age, 'Age')
    # FPFN_NF_Covid(TrueWithMeta, pred, diseases, Sex, 'Sex')

    df = df.merge(TrueWithMeta_df, left_on="Pid", right_on="Pid")

    FP_total = []
    NNF_total = []

    FN_total = []
    PNF_total = []

    if category_name == 'Sex':
        FPR_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'Age':
        FPR_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP and FNR and #of negative and #of positive NF in Covid")

    for c in category:  # 遍历sex和age两种情况下，各个值
        FP_y = []
        NNF_y = []

        FN_y = []
        PNF_y = []

        for d in diseases:
            pred_disease = "bi_" + d

            # number of patient in subgroup with actual NF=0        covid真实值为0的，无病的
            gt_fp = df.loc[(df[d] == 0) & (df[category_name] == c), :]

            # number of patient in subgroup with actual NF=1
            gt_fn = df.loc[(df[d] == 1) & (df[category_name] == c), :]

            # number of patient in subgroup with actual NF=0 and pred NF=1
            pred_fp = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]

            # number of patient in subgroup with actual NF=1 and pred NF=0 FNR
            pred_fn = df.loc[(df[pred_disease] == 0) & (df[d] == 1) & (df[category_name] == c), :]

            if len(gt_fp) != 0:
                FPR = len(pred_fp) / len(gt_fp)
                # number of actual NF = 0
                Percentage = len(gt_fp)

                FP_y.append(FPR)
                NNF_y.append(Percentage)
            else:
                FP_y.append(np.NaN)
                NNF_y.append(0)

            if len(gt_fn) != 0:
                FNR = len(pred_fn) / len(gt_fn)
                # number of actual NF = 1
                Percentage = len(gt_fn)

                FN_y.append(FNR)
                PNF_y.append(Percentage)

            else:
                FN_y.append(np.NaN)
                PNF_y.append(0)

        FP_total.append(FP_y)
        NNF_total.append(NNF_y)

        FN_total.append(FN_y)
        PNF_total.append(PNF_y)

    for i in range(len(FN_total)):

        if category_name == 'Sex':
            if i == 0:
                Perc_S = pd.DataFrame(NNF_total[i], columns=["#NNF_M"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                Perc_S = pd.DataFrame(PNF_total[i], columns=["#PNF_M"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_M"])
                FPR_sex = pd.concat([FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)

                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_M"])
                FPR_sex = pd.concat([FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)

            if i == 1:
                Perc_S = pd.DataFrame(NNF_total[i], columns=["#NNF_F"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                Perc_S = pd.DataFrame(PNF_total[i], columns=["#PNF_F"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_F"])
                FPR_sex = pd.concat([FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)

                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_F"])
                FPR_sex = pd.concat([FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)

            FPR_sex.to_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/FPRFNR_NF_sex.csv")
            # 列名为：    ,diseases,#NNF_M,#PNF_M,FPR_M,FNR_M,#NNF_F,#PNF_F,FPR_F,FNR_F

        if category_name == 'Age':
            if i == 0:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_40-60"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_40-60"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_40-60"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_40-60"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 1:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_60-80"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_60-80"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_60-80"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_60-80"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 2:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_20-40"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_20-40"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_20-40"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_20-40"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 3:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_80-"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_80-"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_80-"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_80-"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 4:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_0-20"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_0-20"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_0-20"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_0-20"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)
            FPR_age.to_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/FPRFNR_NF_age.csv")


'''
列名为：    ,diseases,#NNF_40-60,#PNF_40-60,FPR_40-60,FNR_40-60,#NNF_60-80,#PNF_60-80,FPR_60-80,FNR_60-80,
                     #NNF_20-40,#PNF_20-40,FPR_20-40,FNR_20-40,#NNF_80-,#PNF_80-,FPR_80-,FNR_80-,#NNF_0-20,#PNF_0-20,FPR_0-20,FNR_0-20
'''


def FPFN_NF_Covid_Inter(TrueWithMeta_df, df, diseases, category1, category_name1, category2, category_name2):
    #  return FPR and FNR for the 'No Finding' per intersection
    # 这是最后传入的参数，可以参考↓
    # FPFN_NF_Covid_Inter(TrueWithMeta, pred, diseases, Sex, 'Sex', Age, 'Age')

    df = df.merge(TrueWithMeta_df, left_on="Pid", right_on="Pid")

    if (category_name1 == 'Sex') & (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["SexAge"])

    print("Intersectional identity FPR and FNR")
    i = 0

    # 性别
    for c1 in range(len(category1)):
        FPR_list = []
        FNR_list = []

        # 年龄
        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                pred_disease = "bi_" + diseases[d]

                # NF=0 Sex Age
                gt_fp = df.loc[((df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (
                        df[category_name2] == category2[c2])), :]

                # bi_NF=1 NF=0 Sex Age
                pred_fp = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (
                        df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]

                # NF=1 Sex Age
                gt_fn = df.loc[((df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) & (
                        df[category_name2] == category2[c2])), :]

                # bi_NF=0 NF=1 Sex Age
                pred_fn = df.loc[((df[pred_disease] == 0) & (df[diseases[d]] == 1) & (
                        df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]

                if len(gt_fp) != 0:
                    FPR = len(pred_fp) / len(gt_fp)
                else:
                    FPR = np.NaN

                if len(gt_fn) != 0:
                    FNR = len(pred_fn) / len(gt_fn)
                else:
                    FNR = np.NaN

            FPR_list.append(FPR)
            FNR_list.append(FNR)

        if (category_name1 == 'Sex') & (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

                FPR_SA = pd.DataFrame(FNR_list, columns=["FNR_M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

                FPR_SA = pd.DataFrame(FNR_list, columns=["FNR_F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

        i = i + 1

    if (category_name1 == 'Sex') & (category_name2 == 'Age'):
        FP_AgeSex.to_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/FPFN_AgeSex.csv")


def FP_NF_Covid_MEMBERSHIP_Num_Inter0(TrueWithMeta_df, df, diseases, category1, category_name1, category2,
                                      category_name2):
    df = df.merge(TrueWithMeta_df, left_on="Pid", right_on="Pid")

    if (category_name1 == 'Sex') & (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])

    print("Number of patient with actual NF=0")
    i = 0

    # 性别
    for c1 in range(len(category1)):
        FPR_list = []

        # 年龄
        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                # Number of patient with actual NF=0 Sex Age
                SubDivision = df.loc[((df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (
                        df[category_name2] == category2[c2])), :]

            FPR_list.append(len(SubDivision))

        if (category_name1 == 'Sex') & (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

        i = i + 1

    if (category_name1 == 'Sex') & (category_name2 == 'Age'):
        FP_AgeSex.to_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/Num_NNF_AgeSex.csv")  # 所有没病的。0


# ------------------------------------------------------------------------
# FN, NF. Membership count with actual NF =1
# ------------------------------------------------------------------------

def FN_NF_Covid_MEMBERSHIP_Num_Inter1(TrueWithMeta_df, df, diseases, category1, category_name1, category2,
                                      category_name2):
    df = df.merge(TrueWithMeta_df, left_on="Pid", right_on="Pid")

    if (category_name1 == 'Sex') & (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])

    print("Number of patient with actual NF=1")
    i = 0
    for c1 in range(len(category1)):
        NNF_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                # Nubmer of patient within intersection with actual NF = 1
                SubDivision = df.loc[((df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) & (
                        df[category_name2] == category2[c2])), :]

            NNF_list.append(len(SubDivision))

        if (category_name1 == 'Sex') & (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(NNF_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(NNF_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

        i = i + 1

    # Number of patients with actual No Finding = 1 within the intersection 
    if (category_name1 == 'Sex') & (category_name2 == 'Age'):
        FP_AgeSex.to_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/Num_PNF_AgeSex.csv")


def FPR_Underdiagnosis():
    # MIMIC data
    # diseases = ['No Finding']
    diseases = ['COVID-19']
    Age = ['40-60', '60-80', '20-40', '80-', '0-20']
    Sex = ['M', 'F']

    pred = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/bipred.csv")
    TrueWithMeta = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/True_withMeta.csv")

    factor_CXP = [Sex, Age]
    factor_str_CXP = ['Sex', 'Age']

    # --------------------------------------------------------------
    #     #Subgroup-specific Chronic Underdiagnosis
    # --------------------------------------------------------------

    # 1.subgroup下 NF=0 和 NF=1 的数量
    # 2. subgroup下 FPR 和 FNR
    FPFN_NF_Covid(TrueWithMeta, pred, diseases, Age, 'Age')
    FPFN_NF_Covid(TrueWithMeta, pred, diseases, Sex, 'Sex')

    # --------------------------------------------------------------
    #     #Intersectional-specific Chronic Underdiagnosis交叉组
    # --------------------------------------------------------------
    # Sex Age 交叉 FPR FNR
    FPFN_NF_Covid_Inter(TrueWithMeta, pred, diseases, Sex, 'Sex', Age, 'Age')

    # --------------------------------------------------------------
    #     #Intersectional Membership FN_NF_CXP_MEMBERSHIP_Num_Inter
    # --------------------------------------------------------------

    # NF=0 Age Sex的数量
    FP_NF_Covid_MEMBERSHIP_Num_Inter0(TrueWithMeta, pred, diseases, Sex, 'Sex', Age, 'Age')

    # NF=1 Age Sex的数量
    FN_NF_Covid_MEMBERSHIP_Num_Inter1(TrueWithMeta, pred, diseases, Sex, 'Sex', Age, 'Age')


def preprocess():
    testTrue = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results5/testTrue.csv")
    new_test = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/metadata_test.csv')

    metaData = {'Pid': new_test['Pid'], 'Sex': new_test['Sex'], 'Age': new_test['Age']}
    metaData = pd.DataFrame(metaData)

    True_withMeta = testTrue.merge(metaData, left_on="Pid", right_on="Pid")
    True_withMeta['Age'] = np.where(True_withMeta['Age'].between(0, 19), 19, True_withMeta['Age'])
    True_withMeta['Age'] = np.where(True_withMeta['Age'].between(20, 39), 39, True_withMeta['Age'])
    True_withMeta['Age'] = np.where(True_withMeta['Age'].between(40, 59), 59, True_withMeta['Age'])
    True_withMeta['Age'] = np.where(True_withMeta['Age'].between(60, 79), 79, True_withMeta['Age'])
    True_withMeta['Age'] = np.where(True_withMeta['Age'] >= 80, 81, True_withMeta['Age'])
    True_withMeta = True_withMeta.replace([[None], [np.NaN], 19, 39, 59, 79, 81, 'Male', 'Female'],
                                          [0, 0, "0-20", "20-40", "40-60", "60-80", "80-", 'M', 'F'])

    True_withMeta.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/True_withMeta.csv')


if __name__ == '__main__':
    preprocess()
    FPR_Underdiagnosis()
