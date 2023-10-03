import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


# # 5个模型 5个年龄段平均#NNF,FPR,CI_FPR,FNR,CI_FNR
# FP5_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPRFNR_NF_age.csv")
# FP4_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPRFNR_NF_age.csv")
# FP3_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPRFNR_NF_age.csv")
# FP2_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPRFNR_NF_age.csv")
# FP1_age = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPRFNR_NF_age.csv")
# FP_age = pd.concat([FP1_age, FP2_age, FP3_age, FP4_age, FP5_age], axis=0, ignore_index=True)
#
# F_age = pd.DataFrame({'diseases': FP_age["diseases"], '40-60': FP_age["FPR_40-60"], '60-80': FP_age["FPR_60-80"],
#                       '20-40': FP_age["FPR_20-40"], '80-': FP_age["FPR_80-"], '0-20': FP_age["FPR_0-20"]})
# F_age_df = F_age.describe()
# print("FPR distribiution in ages")
# print('mean of FPR over age stages', round(F_age_df.loc['mean']))
# print('95% CI of FPR over age stages', round(1.96 * F_age_df.loc['std'] / np.sqrt(5)))  # 95%置信区间的FPR(每个年龄段都有一个值)
# print("Covid: Mean FPR distribiution over ages", mean(F_age_df.loc['mean']))
#
#
# FP_age_df = FP_age.describe()
# Age = ['0-20', '20-40', '40-60', '60-80', '80-']
# FPR_FNR_Age_df = pd.DataFrame(Age, columns=["Age"])
#
#
# # FPR_FNR_Age_df = FiveRunSubgroup(Age, FP_age_df, FPR_FNR_Age_df)↓
# def FiveRunSubgroup(factors, df_in, df_out):
#     FPR = []
#     FNR = []
#     NNF = []
#     PNF = []
#     CI_FPR = []
#     CI_FNR = []
#     ConfI = 1.96 * df_in.loc['std'] / np.sqrt(5)
#     if factors == Age:
#         for fact in factors:
#             NNF.append(df_in.loc['mean']['#NNF_' + fact])
#             PNF.append(df_in.loc['mean']['#PNF_' + fact])
#             FPR.append(round(df_in.loc['mean']['FPR_' + fact], 3))
#             FNR.append(round(df_in.loc['mean']['FNR_' + fact], 3))
#             CI_FPR.append(round(ConfI.loc['FPR_' + fact], 3))
#             CI_FNR.append(round(ConfI.loc['FNR_' + fact], 3))
#
#     elif factors == Sex:
#         for fact in factors:
#             NNF.append(df_in.loc['mean']['#NNF_' + fact])
#             PNF.append(df_in.loc['mean']['#PNF_' + fact])
#             FPR.append(round(df_in.loc['mean']['FPR_' + fact], 3))
#             FNR.append(round(df_in.loc['mean']['FNR_' + fact], 3))
#             CI_FPR.append(round(ConfI.loc['FPR_' + fact], 3))
#             CI_FNR.append(round(ConfI.loc['FNR_' + fact], 3))
#
#     elif factors == Country:
#         for fact in factors:
#             NNF.append(df_in.loc['mean']['#NNF_Country_' + fact])
#             PNF.append(df_in.loc['mean']['#PNF_Country_' + fact])
#             FPR.append(round(df_in.loc['mean']['FPR_Country_' + fact], 3))
#             FNR.append(round(df_in.loc['mean']['FNR_Country_' + fact], 3))
#             CI_FPR.append(round(ConfI.loc['FPR_Country_' + fact], 3))
#             CI_FNR.append(round(ConfI.loc['FNR_Country_' + fact], 3))
#
#     # else:
#     #     print("出错了")
#
#     df_out['#NNF'] = NNF
#     df_out['#PNF'] = PNF
#     df_out['FPR'] = FPR
#     df_out['CI_FPR'] = CI_FPR
#
#     df_out['FNR'] = FNR
#     df_out['CI_FNR'] = CI_FNR
#
#     return df_out
#
#
# FPR_FNR_Age_df = FiveRunSubgroup(Age, FP_age_df, FPR_FNR_Age_df)
# FPR_FNR_Age_df.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Subgroun_FNR_FPR_Age.csv')
#
# # 5个模型 2个性别平均 #NNF, FPR, CI_FPR, FNR, CI_FNR
# FP5_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPRFNR_NF_sex.csv")
# FP4_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPRFNR_NF_sex.csv")
# FP3_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPRFNR_NF_sex.csv")
# FP2_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPRFNR_NF_sex.csv")
# FP1_sex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPRFNR_NF_sex.csv")
# FP_sex = pd.concat([FP1_sex, FP2_sex, FP3_sex, FP4_sex, FP5_sex], axis=0, ignore_index=True)
#
# F_sex = pd.DataFrame({'diseases': FP_sex["diseases"], 'M': FP_sex["FPR_M"], 'F': FP_sex["FPR_F"]})
# F_sex_df = F_sex.describe()
# print("FPR mean distribiution in sexes", F_sex_df.loc['mean'])
# print("FPR distribiution in sexes Confidence interval", 1.96 * F_sex_df.loc['std'] / np.sqrt(5))
# print("CXP: Mean FPR distribiution over sexes", mean(F_sex_df.loc['mean']))
#
# FP_sex_df = FP_sex.describe()
# Sex = ['M', 'F']
# FPR_FNR_sex_df = pd.DataFrame(Sex, columns=["sex"])
#
# FPR_FNR_sex_df = FiveRunSubgroup(Sex, FP_sex_df, FPR_FNR_sex_df)
# FPR_FNR_sex_df.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Subgroun_FNR_FPR_Sex.csv')
#
#
# # 5个模型 2个Country平均 #NNF, FPR, CI_FPR, FNR, CI_FNR
# FP5_Country = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPRFNR_NF_Country.csv")
# FP4_Country = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPRFNR_NF_Country.csv")
# FP3_Country = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPRFNR_NF_Country.csv")
# FP2_Country = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPRFNR_NF_Country.csv")
# FP1_Country = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPRFNR_NF_Country.csv")
# FP_Country = pd.concat([FP1_Country, FP2_Country, FP3_Country, FP4_Country, FP5_Country], axis=0, ignore_index=True)
#
# F_Country = pd.DataFrame({'diseases': FP_Country["diseases"], 'France': FP_Country["FPR_Country_France"], 'China': FP_Country["FPR_Country_China"], 'Iran': FP_Country["FPR_Country_Iran"], 'OTHER': FP_Country["FPR_Country_OTHER"]})
# F_Country_df = F_Country.describe()
# print("FPR mean distribiution in Country", F_Country_df.loc['mean'])
# print("FPR distribiution in Country Confidence interval", 1.96 * F_Country_df.loc['std'] / np.sqrt(5))
# print("CXP: Mean FPR distribiution over Country", mean(F_Country_df.loc['mean']))
#
# FP_Country_df = FP_Country.describe()
# Country = ['France', 'China', 'Iran', 'OTHER']
# FPR_FNR_Country_df = pd.DataFrame(Country, columns=["Country"])
#
# FPR_FNR_Country_df = FiveRunSubgroup(Country, FP_Country_df, FPR_FNR_Country_df)
# FPR_FNR_Country_df.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Subgroun_FNR_FPR_Country.csv')











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

    want_df = want_df.merge(PNF_df, left_on="Country", right_on="Country")
    want_df = want_df.merge(NNF_df, left_on="Country", right_on="Country")

    return want_df

# 年龄性别交叉平均 FPR CI Num_NNF
FP5_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPFN_AgeSex.csv")
FP4_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPFN_AgeSex.csv")
FP3_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPFN_AgeSex.csv")
FP2_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPFN_AgeSex.csv")
FP1_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPFN_AgeSex.csv")
FP_agesex = pd.concat([FP1_agesex, FP2_agesex, FP3_agesex, FP4_agesex, FP5_agesex], axis=0, ignore_index=True)

FP_AgeSex = FP_agesex.groupby("SexAge")
FP_AgSx_df = FP_AgeSex.describe()
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
Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_PNF_AgeSex.csv")   # 5个里面的这个文件内容都一样
Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_NNF_AgeSex.csv")

want = FiveRun(factors1, factors2, AgeSex_df, FP_AgSx_df, Num_PNF_df, Num_NNF_df)
want.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_AgeSex.csv')



# Country性别交叉平均 FPR CI Num_NNF
FP5_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPFN_CountrySex.csv")
FP4_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPFN_CountrySex.csv")
FP3_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPFN_CountrySex.csv")
FP2_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPFN_CountrySex.csv")
FP1_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPFN_CountrySex.csv")
FP_Countrysex = pd.concat([FP1_Countrysex, FP2_Countrysex, FP3_Countrysex, FP4_Countrysex, FP5_Countrysex], axis=0, ignore_index=True)

FP_CountrySex = FP_Countrysex.groupby("SexCountry")
FP_RacSx_df = FP_CountrySex.describe()

factors1 = ['FPR_M', 'FNR_M', 'FPR_F', 'FNR_F']
factors2 = ['M', 'F']
Country = ['France', 'China', 'Iran', 'OTHER']
CountrySex_df = pd.DataFrame(Country, columns=["Country"])
Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_PNF_CountrySex.csv")   # 5个里面的这个文件内容都一样
Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_NNF_CountrySex.csv")
want = FiveRun(factors1, factors2, CountrySex_df, FP_RacSx_df, Num_PNF_df, Num_NNF_df)
want.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_CountrySex.csv')
