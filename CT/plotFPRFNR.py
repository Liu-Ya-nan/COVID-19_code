# @Time: 2022/6/1 18:57
# @Author: lynn
# @File: plotFRPFNR.py
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # # number_positive, number_negative, number_pred_fp, number_pred_fn 180 76 73 0
# # # number_positive, number_negative, number_pred_fp, number_pred_fn 0 0 0 0
# # # number_positive, number_negative, number_pred_fp, number_pred_fn 52 30 30 2
# # # number_positive, number_negative, number_pred_fp, number_pred_fn 0 4 4 0
# # # number_positive, number_negative, number_pred_fp, number_pred_fn 32 0 0 0
# # d
#
#
# FPR_FNR_sex_df = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Subgroun_FNR_FPR_Sex.csv')
# FPR_FNR_Age_df = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Subgroun_FNR_FPR_Age.csv')
#
#
# def PlotSubgroup(Param, my_colors, NotEnough=False):
#     fig, ax = plt.subplots(figsize=(3, 2))
#     fontsize = 10
#
#     sex = ('Male', 'Female', '', '80-', '60-80', '40-60', '20-40', '0-20')
#     sex_pos = np.arange(len(sex))
#
#     FPR = (FPR_FNR_sex_df.loc[FPR_FNR_sex_df['sex'] == 'M', Param].tolist()[0],
#            FPR_FNR_sex_df.loc[FPR_FNR_sex_df['sex'] == 'F', Param].tolist()[0],
#            np.NaN,
#            FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '80-', Param].tolist()[0],
#            FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '60-80', Param].tolist()[0],
#            FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '40-60', Param].tolist()[0],
#            FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '20-40', Param].tolist()[0],
#            FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '0-20', Param].tolist()[0])
#
#     error = (FPR_FNR_sex_df.loc[FPR_FNR_sex_df['sex'] == 'M', 'CI_' + Param].tolist()[0],
#              FPR_FNR_sex_df.loc[FPR_FNR_sex_df['sex'] == 'F', 'CI_' + Param].tolist()[0],
#              np.NaN,
#              FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '80-', 'CI_' + Param].tolist()[0],
#              FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '60-80', 'CI_' + Param].tolist()[0],
#              FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '40-60', 'CI_' + Param].tolist()[0],
#              FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '20-40', 'CI_' + Param].tolist()[0],
#              FPR_FNR_Age_df.loc[FPR_FNR_Age_df['Age'] == '0-20', 'CI_' + Param].tolist()[0])
#
#     print(FPR)
#     print(error)
#     # (0.094, 0.089, nan, 0.02, 0.063, 0.125, 0.224, 0.116)
#     # (0.006, 0.007, nan, 0.003, 0.007, 0.007, 0.008, 0.012)
#     # (0.369, 0.337, nan, 0.823, 0.54, 0.296, 0.174, 0.133)
#     # (0.023, 0.017, nan, 0.027, 0.032, 0.016, 0.016, 0.025)
#
#     # (0.077, 0.074, nan, 0.014, 0.053, 0.104, 0.162, 0.235)
#     # (0.004, 0.004, nan, 0.004, 0.003, 0.005, 0.007, 0.021)
#     # (0.39, 0.354, nan, 0.819, 0.559, 0.317, 0.199, 0.216)
#     # (0.015, 0.014, nan, 0.042, 0.019, 0.014, 0.017, 0.034)
#
#     color = ['pink', 'pink', 'white', 'green', 'green', 'green', 'green', 'green']
#
#     # ax.bar(sex_pos, FPR, yerr=error, align='center', color=color)
#     ax.bar(sex_pos, FPR, align='center', color=color)
#
#     labels = ['Male', 'Female', '', '80-', '60-80', '40-60', '20-40', '0-20']
#     x_pos = np.arange(len(labels))
#     #  y_labels = ['0.0','0.1', '0.2','0.3', '0.4','0.5', '0.6','0.7','0.8']
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
#     ax.set_title(Param, fontsize = fontsize)
#     # y轴最高0.8
#     # ax.set_ylim(top=1.0)
#     ax.set_ylim([0, 0.05])
#     ax.yaxis.grid(True)
#
#     if NotEnough:
#
#         # my_colors = ['','grey', 'k','k', 'k','k']
#
#         for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
#             ticklabel.set_color(tickcolor)
#
#     plt.savefig('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/FPRFNR/' + Param + '_NF.jpg', bbox_inches='tight', dpi=1000)
#
#
# PlotSubgroup('FPR', my_colors=[])
# PlotSubgroup('FNR', my_colors=[])


want = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_AgeSex.csv')

# Intersectional FNR FPR
def plotInterFNR_FPR(param):
    plt.rcdefaults()

    fig, ax = plt.subplots(figsize=(2, 2))
    fontsize = 10

    # age
    age = ('80-', '60-80', '40-60', '20-40', '0-20')
    age_pos = np.arange(len(age))
    FNR = (want.loc[want['Age'] == '80-', param].tolist()[0],
           want.loc[want['Age'] == '60-80', param].tolist()[0],
           want.loc[want['Age'] == '40-60', param].tolist()[0],
           want.loc[want['Age'] == '20-40', param].tolist()[0],
           want.loc[want['Age'] == '0-20', param].tolist()[0]
    )

    error = (want.loc[want['Age'] == '80-', 'CI_' + param].tolist()[0],
             want.loc[want['Age'] == '60-80', 'CI_' + param].tolist()[0],
             want.loc[want['Age'] == '40-60', 'CI_' + param].tolist()[0],
             want.loc[want['Age'] == '20-40', 'CI_' + param].tolist()[0],
             want.loc[want['Age'] == '0-20', 'CI_' + param].tolist()[0],

    )

    color = ['green', 'green', 'green', 'green', 'green']

    ax.bar(age_pos, FNR, yerr=error, align='center', color=color)

    labels = ['80-', '60-80', '40-60', '20-40', '0-20']
    x_pos = np.arange(len(labels))
    # y_labels = ['0.0', '0.2', '0.4', '0.6','0.8']
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
    # ax.set_yticklabels(y_labels, fontsize = fontsize)
    # ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_ylabel('FNR',fontsize = fontsize)
    ax.set_title(param, fontsize=fontsize)
    ax.set_ylim([0, 0.05])

    ax.yaxis.grid(True)

    # my_colors = ['k', 'k', 'k', 'k', 'grey']
    #
    # for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
    #     ticklabel.set_color(tickcolor)

    plt.savefig('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/FPRFNR/Int_' + param + '12345111.jpg', bbox_inches='tight')

plotInterFNR_FPR('FNR_F')
# plotInterFNR_FPR('FNR_M')
# plotInterFNR_FPR('FPR_F')
# plotInterFNR_FPR('FPR_M')

