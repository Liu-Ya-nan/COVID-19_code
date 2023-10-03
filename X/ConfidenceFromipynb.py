import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

# # 1.计算5个模型平均AUC 95%置信区间
# # 2.计算每个疾病在5个模型上的平均AUC 95%置信区间
# all_df_path = '/home/user1/data/lyn/covid-chestxray/CheXpert-v1.0-small/all.csv'
#
# WholeData = pd.read_csv(all_df_path)
# print("Number of images:", len(WholeData))  # 223648图片
#
# path = WholeData['Path']
# patient = path.str.split('/')
#
# # 添加新列subject_id
# subject_id = []
# for i in range(len(patient)):
#     id = int(patient[i][2][7:])
#     subject_id.append(id)
# WholeData['subject_id'] = subject_id
# print(WholeData)
#
# Whole = WholeData.groupby("subject_id")
# print("Number of unique patients:", len(Whole.count()))  # 64740病人
#
# WholeDataX = WholeData.groupby("Sex")
# # describe()描述组内数据的基本统计量
# # 只有数字类型的列数据才会计算统计
# WholeDataX_df = WholeDataX.describe()
#
# dfWhole_Sex = WholeDataX_df["subject_id"]['count']  # (Female, 90883)(Male, 132764)(Unknown, 1)
# total_CXP = dfWhole_Sex["Female"] + dfWhole_Sex["Male"]  # 23647
# print("Male Percent:  ", 100 * dfWhole_Sex["Male"] / total_CXP)  # 59.36%
# print("female Percent:  ", 100 * dfWhole_Sex["Female"] / total_CXP)  # 40.64%
# print("#images", total_CXP)  # 23647
#
# # WholeData_SEX = WholeData['Sex'].value_counts()     # (Female, 90883)(Male, 132764)(Unknown, 1)
# # total_CXP = WholeData_SEX['Female'] + WholeData_SEX['Male']
#
# WholeData_020 = WholeData[(WholeData['Age'] >= 0) & (WholeData['Age'] <= 19)]  # 1940
# WholeData_2040 = WholeData[(WholeData['Age'] >= 20) & (WholeData['Age'] <= 39)]  # 29474
# WholeData_4060 = WholeData[(WholeData['Age'] >= 40) & (WholeData['Age'] <= 59)]  # 69343
# WholeData_6080 = WholeData[(WholeData['Age'] >= 60) & (WholeData['Age'] <= 79)]  # 87083
# WholeData_80 = WholeData[(WholeData['Age'] >= 80)]  # 35808
# # 223648
# totalAgeCXP = len(WholeData_020) + len(WholeData_2040) + len(WholeData_4060) + len(WholeData_6080) + len(WholeData_80)
#
# print("'0-20' Percent:", 100 * len(WholeData_020) / totalAgeCXP)  # 0.8674345399914151
# print("'20-40' Percent:", 100 * len(WholeData_2040) / totalAgeCXP)  # 13.178745170982973
# print("'40-60' Percent:", 100 * len(WholeData_4060) / totalAgeCXP)  # 31.005419230218916
# print("'60-80' Percent:", 100 * len(WholeData_6080) / totalAgeCXP)  # 38.93752682787237
# print("'80-' Percent:", 100 * len(WholeData_80) / totalAgeCXP)  # 16.010874230934327
#
# AUC over 5 runs
# Eval5 = pd.read_csv("/home/user1/data/lyn/covid-chestxray/results85/Eval5.csv")
# Eval4 = pd.read_csv("/home/user1/data/lyn/covid-chestxray/results69/Eval4.csv")
# Eval3 = pd.read_csv("/home/user1/data/lyn/covid-chestxray/results64/Eval3.csv")
# Eval2 = pd.read_csv("/home/user1/data/lyn/covid-chestxray/results34/Eval2.csv")
# Eval1 = pd.read_csv("/home/user1/data/lyn/covid-chestxray/results18/Eval1.csv")
#
# Evalall = {"auc1": Eval1['auc'], "auc2": Eval2['auc'], "auc3": Eval3['auc'], "auc4": Eval4['auc'], "auc5": Eval5['auc']}
# Evalall = pd.DataFrame(Evalall)
# print('AUC over 5 run', Evalall)
# # 每一个模型在14个标签上AUC的平均值
# # (auc: 0.7975222599892912)(auc2: 0.7972632164073536)(auc3: 0.796896808006766)(auc4: 0.8023333633662294)(auc5: 0.8046332356618902)
# Evalmean = Evalall.mean(axis=0)
#
# # # 95% 的置信区间[u-1.96*标准差/根号n, u+1.96*标准差/根号n]
# # print("CXP Mean of 14 aucs mean over 5 run:", round(Evalall.mean(axis=0).mean(), 3))        # 0.8
# # # 0.003 置信区间：0.8+-0.003
# # print("CXP Confidence interval of 14 aucs mean over 5 run:", round(1.96 * Evalall.mean(axis=0).std() / np.sqrt(5), 3))
#
# # 百分位数是统计中使用的度量，表示小于这个值的观察值的百分比
# print('置信区间上限：', round(np.percentile(Evalmean, 2.5), 3))
# print('置信区间下限：', round(np.percentile(Evalmean, 97.5), 3))
#
# # print("CXP Mean of auce per disease over 5 run:", round(Evalall.mean(axis=1), 3))
# # print("CXP Confidence interval of auce per disease over 5 run:", round(1.96 * Evalall.std(axis=1) / np.sqrt(5), 3))
#
# print(np.percentile(Evalall, 2.5, axis=1))
# print(np.percentile(Evalall, 97.5, axis=1))

# 5个模型 5个年龄段平均#NNF,FPR,CI_FPR,FNR,CI_FNR
FP5_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results19/FPRFNR_NF_age.csv")
FP4_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results21/FPRFNR_NF_age.csv")
FP3_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results35/FPRFNR_NF_age.csv")
FP2_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results51/FPRFNR_NF_age.csv")
FP1_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results70/FPRFNR_NF_age.csv")
FP_age = pd.concat([FP1_age, FP2_age, FP3_age, FP4_age, FP5_age], axis=0, ignore_index=True)

F_age = pd.DataFrame({'diseases': FP_age["diseases"], '40-60': FP_age["FPR_40-60"], '60-80': FP_age["FPR_60-80"],
                      '20-40': FP_age["FPR_20-40"], '80-': FP_age["FPR_80-"], '0-20': FP_age["FPR_0-20"]})
F_age_df = F_age.describe()
print(F_age_df)
# print(F_age)
print("FPR distribiution in ages")
print('mean of FPR over age stages', round(F_age_df.loc['mean']))
print('95% CI of FPR over age stages', round(1.96 * F_age_df.loc['std'] / np.sqrt(5)))  # 95%置信区间的FPR(每个年龄段都有一个值)
print("Covid: Mean FPR distribiution over ages", mean(F_age_df.loc['mean']))


FP_age_df = FP_age.describe()
print(FP_age_df)
Age = ['0-20', '20-40', '40-60', '60-80', '80-']
FPR_FNR_Age_df = pd.DataFrame(Age, columns=["Age"])


# FPR_FNR_Age_df = FiveRunSubgroup(Age, FP_age_df, FPR_FNR_Age_df)↓
def FiveRunSubgroup(factors, df_in, df_out):
    FPR = []
    FNR = []
    NNF = []
    PNF = []
    CI_FPR = []
    CI_FNR = []
    ConfI = 1.96 * df_in.loc['std'] / np.sqrt(5)

    for fact in factors:
        NNF.append(df_in.loc['mean']['#NNF_' + fact])
        PNF.append(df_in.loc['mean']['#PNF_' + fact])
        FPR.append(round(df_in.loc['mean']['FPR_' + fact], 3))
        FNR.append(round(df_in.loc['mean']['FNR_' + fact], 3))
        CI_FPR.append(round(ConfI.loc['FPR_' + fact], 3))
        CI_FNR.append(round(ConfI.loc['FNR_' + fact], 3))

    df_out['#NNF'] = NNF
    df_out['#PNF'] = PNF
    df_out['FPR'] = FPR
    df_out['CI_FPR'] = CI_FPR

    df_out['FNR'] = FNR
    df_out['CI_FNR'] = CI_FNR

    return df_out


FPR_FNR_Age_df = FiveRunSubgroup(Age, FP_age_df, FPR_FNR_Age_df)
FPR_FNR_Age_df.to_csv('/home/qwe/data/cuihaihang/lyn/X2/results/Subgroun_FNR_FPR_Age.csv')

# 5个模型 2个性别平均 #NNF, FPR, CI_FPR, FNR, CI_FNR
FP5_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results19/FPRFNR_NF_sex.csv")
FP4_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results21/FPRFNR_NF_sex.csv")
FP3_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results35/FPRFNR_NF_sex.csv")
FP2_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results51/FPRFNR_NF_sex.csv")
FP1_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results70/FPRFNR_NF_sex.csv")
FP_sex = pd.concat([FP1_sex, FP2_sex, FP3_sex, FP4_sex, FP5_sex], axis=0, ignore_index=True)

F_sex = pd.DataFrame({'diseases': FP_sex["diseases"], 'M': FP_sex["FPR_M"], 'F': FP_sex["FPR_F"]})
F_sex_df = F_sex.describe()
print(F_sex_df)
print("FPR mean distribiution in sexes", F_sex_df.loc['mean'])
print("FPR distribiution in sexes Confidence interval", 1.96 * F_sex_df.loc['std'] / np.sqrt(5))
print("CXP: Mean FPR distribiution over sexes", mean(F_sex_df.loc['mean']))

FP_sex_df = FP_sex.describe()
print(FP_sex_df)
Sex = ['M', 'F']
FPR_FNR_sex_df = pd.DataFrame(Sex, columns=["sex"])

FPR_FNR_sex_df = FiveRunSubgroup(Sex, FP_sex_df, FPR_FNR_sex_df)
FPR_FNR_sex_df.to_csv('/home/qwe/data/cuihaihang/lyn/X2/results/Subgroun_FNR_FPR_Sex.csv')


# 年龄性别交叉平均 FPR CI Num_NNF
FP5_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results19/FPFN_AgeSex.csv")
FP4_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results21/FPFN_AgeSex.csv")
FP3_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results35/FPFN_AgeSex.csv")
FP2_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results51/FPFN_AgeSex.csv")
FP1_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results70/FPFN_AgeSex.csv")
FP_agesex = pd.concat([FP1_agesex, FP2_agesex, FP3_agesex, FP4_agesex, FP5_agesex], axis=0, ignore_index=True)

FP_AgeSex = FP_agesex.groupby("SexAge")
FP_AgSx_df = FP_AgeSex.describe()
print(FP_AgSx_df)
# print("男性各个年龄段FPR均值", round(FP_AgSx_df['FPR_M']['mean'], 3))
# print("男性FPR平均值:", round(FP_AgSx_df['FPR_M']['mean'], 3).mean())
# print("男性各个年龄段FPR 95%CI", round(1.96 * FP_AgSx_df['FPR_M']["std"] / np.sqrt(5), 3))
#
# print("女性各个年龄段FPR均值", round(FP_AgSx_df['FPR_F']['mean'], 3))
# print("女性FPR平均值:", round(FP_AgSx_df['FPR_F']['mean'], 3).mean())
# print("女性各个年龄段FPR 95%CI", round(1.96 * FP_AgSx_df['FPR_F']["std"] / np.sqrt(5), 3))

factors1 = ['FPR_M', 'FNR_M', 'FPR_F', 'FNR_F']
factors2 = ['M', 'F']
age = ['0-20', '20-40', '40-60', '60-80', '80-']
AgeSex_df = pd.DataFrame(age, columns=["Age"])
Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results19/Num_PNF_AgeSex.csv")   # 5个里面的这个文件内容都一样
Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/X2/results19/Num_NNF_AgeSex.csv")


# want = FiveRun(factors1, factors2, AgeSex_df, FP_AgSx_df, Num_PNF_df, Num_NNF_df)
def FiveRun(factors1, factors2, want_df, df, PNF_df, NNF_df):
    for factor in factors1:
        dfM0 = round(df[factor]['mean'], 3)
        dfM2 = round(1.96 * df[factor]["std"] / np.sqrt(5), 3)
        want_df[factor] = pd.DataFrame(dfM0.values.tolist(), columns=[factor])
        want_df['CI_' + factor] = pd.DataFrame(dfM2.values.tolist(), columns=['CI_' + factor])

    for factor in factors2:
        PNF_df['PNF_' + factor] = PNF_df[factor]
        NNF_df['NNF_' + factor] = NNF_df[factor]

    PNF_df = PNF_df.drop(factors2, axis=1)
    NNF_df = NNF_df.drop(factors2, axis=1)
    want_df = want_df.merge(PNF_df, left_on="Age", right_on="Age")
    want_df = want_df.merge(NNF_df, left_on="Age", right_on="Age")

    return want_df

want = FiveRun(factors1, factors2, AgeSex_df, FP_AgSx_df, Num_PNF_df, Num_NNF_df)
want.to_csv('/home/qwe/data/cuihaihang/lyn/X2/results/Inter_FPFN_AgeSex.csv')

