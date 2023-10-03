# @Time: 2022/6/1 18:57
# @Author: lynn
# @File: plotFRPFNR.py
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


want_Age = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_AgeCountry.csv')
want_Sex = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_SexCountry.csv')

# Intersectional FNR FPR
def plotInterFNR_FPR(param):
    plt.rcdefaults()

    fig, ax = plt.subplots(figsize=(1.6, 2))
    fontsize = 10

    # age
    # Country = ('France', 'China', 'Iran', 'OTHER')

    age = ('Male', 'Female', '', '80-', '60-80', '40-60', '20-40', '0-20')
    age_pos = np.arange(len(age))
    FNR = (want_Sex.loc[want_Sex['Sex'] == 'M', param].tolist()[0],
           want_Sex.loc[want_Sex['Sex'] == 'F', param].tolist()[0],
           np.NaN,
           want_Age.loc[want_Age['Age'] == '80-', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '60-80', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '40-60', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '20-40', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '0-20', param].tolist()[0])

    # error = (want.loc[want['Age'] == '80-', 'CI_' + param].tolist()[0],
    # )

    color = ['pink', 'pink', 'white', 'green', 'green', 'green', 'green', 'green']

    # ax.bar(age_pos, FNR, yerr=error, align='center', color=color)
    ax.bar(age_pos, FNR, align='center', color=color)

    labels = ['Male', 'Female', '', '80-', '60-80', '40-60', '20-40', '0-20']


    x_pos = np.arange(len(labels))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
    # ax.set_title(param, fontsize=fontsize)
    ax.set_title('Iran', fontsize=fontsize)
    # ax.set_ylim(top=1.0)
    ax.set_ylim([0, 0.07])

    ax.yaxis.grid(True)

    plt.savefig('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/FPRFNR/' + param + '.jpg', bbox_inches='tight', dpi=1000)

plotInterFNR_FPR('FNR_Iran')
# plotInterFNR_FPR('FNR_M')
# plotInterFNR_FPR('FPR_F')
# plotInterFNR_FPR('FPR_M')

