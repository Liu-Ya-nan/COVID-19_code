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

# Sex国家交叉平均 FPR CI Num_NNF
FP5_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPFN_SexCountry.csv")
FP4_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPFN_SexCountry.csv")
FP3_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPFN_SexCountry.csv")
FP2_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPFN_SexCountry.csv")
FP1_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPFN_SexCountry.csv")
FP_Countrysex = pd.concat([FP1_Countrysex, FP2_Countrysex, FP3_Countrysex, FP4_Countrysex, FP5_Countrysex], axis=0, ignore_index=True)

FP_CountrySex = FP_Countrysex.groupby("SexCountry")
FP_AgSx_df = FP_CountrySex.describe()

factors1 = ['FPR_France', 'FNR_France', 'FPR_China', 'FNR_China', 'FPR_Iran', 'FNR_Iran', 'FPR_OTHER', 'FNR_OTHER']
factors2 = ['France', 'China', 'Iran', 'OTHER']
Sex = ['M', 'F']
CountrySex_df = pd.DataFrame(Sex, columns=["Sex"])
Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_PNF_SexCountry.csv")   # 5个里面的这个文件内容都一样
Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_NNF_SexCountry.csv")

want = FiveRun(factors1, factors2, CountrySex_df, FP_AgSx_df, Num_PNF_df, Num_NNF_df)
want.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_SexCountry.csv')



# # Age国家交叉平均 FPR CI Num_NNF
# FP5_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/FPFN_AgeCountry.csv")
# FP4_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet18/FPFN_AgeCountry.csv")
# FP3_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_resnet34/FPFN_AgeCountry.csv")
# FP2_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sfv2/FPFN_AgeCountry.csv")
# FP1_Countrysex = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_sq/FPFN_AgeCountry.csv")
# FP_Countrysex = pd.concat([FP1_Countrysex, FP2_Countrysex, FP3_Countrysex, FP4_Countrysex, FP5_Countrysex], axis=0, ignore_index=True)
#
# FP_CountrySex = FP_Countrysex.groupby("AgeCountry")
# FP_RacSx_df = FP_CountrySex.describe()
#
# factors1 = ['FPR_France', 'FNR_France', 'FPR_China', 'FNR_China', 'FPR_Iran', 'FNR_Iran', 'FPR_OTHER', 'FNR_OTHER']
# factors2 = ['France', 'China', 'Iran', 'OTHER']
# Age = ['0-20', '20-40', '40-60', '60-80', '80-']
# CountrySex_df = pd.DataFrame(Age, columns=["Age"])
# Num_PNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_PNF_AgeCountry.csv")   # 5个里面的这个文件内容都一样
# Num_NNF_df = pd.read_csv("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results_mb/Num_NNF_AgeCountry.csv")
# want = FiveRun(factors1, factors2, CountrySex_df, FP_RacSx_df, Num_PNF_df, Num_NNF_df)
# want.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_AgeCountry.csv')
