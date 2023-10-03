import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean



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

    want_df = want_df.merge(PNF_df, left_on="Sex", right_on="Sex")
    want_df = want_df.merge(NNF_df, left_on="Sex", right_on="Sex")

    return want_df

# 年龄性别交叉平均 FPR CI Num_NNF
FP5_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPFN_SexAge.csv")
FP4_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPFN_SexAge.csv")
FP3_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPFN_SexAge.csv")
FP2_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPFN_SexAge.csv")
FP1_agesex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPFN_SexAge.csv")
FP_agesex = pd.concat([FP1_agesex, FP2_agesex, FP3_agesex, FP4_agesex, FP5_agesex], axis=0, ignore_index=True)

FP_AgeSex = FP_agesex.groupby("AgeSex")
FP_AgSx_df = FP_AgeSex.describe()

factors1 = ['FPR_0-20', 'FNR_0-20', 'FPR_20-40', 'FNR_20-40', 'FPR_40-60', 'FNR_40-60', 'FPR_60-80', 'FNR_60-80', 'FPR_80-', 'FNR_80-']
factors2 = ['0-20', '20-40', '40-60', '60-80', '80-']
Sex = ['M', 'F']
AgeSex_df = pd.DataFrame(Sex, columns=["Sex"])
Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_PNF_SexAge.csv")   # 5个里面的这个文件内容都一样
Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_NNF_SexAge.csv")

want = FiveRun(factors1, factors2, AgeSex_df, FP_AgSx_df, Num_PNF_df, Num_NNF_df)
want.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_SexAge.csv')



# # CountryAge交叉平均 FPR CI Num_NNF
# FP5_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPFN_CountryAge.csv")
# FP4_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPFN_CountryAge.csv")
# FP3_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPFN_CountryAge.csv")
# FP2_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPFN_CountryAge.csv")
# FP1_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPFN_CountryAge.csv")
# FP_Countrysex = pd.concat([FP1_Countrysex, FP2_Countrysex, FP3_Countrysex, FP4_Countrysex, FP5_Countrysex], axis=0, ignore_index=True)
#
# FP_CountrySex = FP_Countrysex.groupby("AgeCountry")
# FP_RacSx_df = FP_CountrySex.describe()
#
# factors1 = ['FPR_0-20', 'FNR_0-20', 'FPR_20-40', 'FNR_20-40', 'FPR_40-60', 'FNR_40-60', 'FPR_60-80', 'FNR_60-80', 'FPR_80-', 'FNR_80-']
# factors2 = ['0-20', '20-40', '40-60', '60-80', '80-']
# Country = ['France', 'China', 'Iran', 'OTHER']
# CountrySex_df = pd.DataFrame(Country, columns=["Country"])
# Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_PNF_CountryAge.csv")   # 5个里面的这个文件内容都一样
# Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_NNF_CountryAge.csv")
# want = FiveRun(factors1, factors2, CountrySex_df, FP_RacSx_df, Num_PNF_df, Num_NNF_df)
# want.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_CountryAge.csv')
