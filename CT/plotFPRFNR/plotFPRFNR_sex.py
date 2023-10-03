# @Time: 2022/6/1 18:57
# @Author: lynn
# @File: plotFRPFNR.py
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


want_Age = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_AgeSex.csv')
want_Country = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/Inter_FPFN_CountrySex.csv')

# Intersectional FNR FPR
def plotInterFNR_FPR(param):
    plt.rcdefaults()

    fig, ax = plt.subplots(figsize=(1.6, 2))
    fontsize = 10

    # age
    # Country = ('France', 'China', 'Iran', 'OTHER')

    age = ('80-', '60-80', '40-60', '20-40', '0-20', '', 'France', 'China', 'Iran', 'OTHER')
    age_pos = np.arange(len(age))
    FNR = (want_Age.loc[want_Age['Age'] == '80-', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '60-80', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '40-60', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '20-40', param].tolist()[0],
           want_Age.loc[want_Age['Age'] == '0-20', param].tolist()[0],
           np.NaN,
           want_Country.loc[want_Country['Country'] == 'France', param].tolist()[0],
           want_Country.loc[want_Country['Country'] == 'China', param].tolist()[0],
           want_Country.loc[want_Country['Country'] == 'Iran', param].tolist()[0],
           want_Country.loc[want_Country['Country'] == 'OTHER', param].tolist()[0]
           )

    # error = (want.loc[want['Age'] == '80-', 'CI_' + param].tolist()[0],
    # )

    color = ['green', 'green', 'green', 'green', 'green', 'white', 'pink', 'pink', 'pink', 'pink']

    # ax.bar(age_pos, FNR, yerr=error, align='center', color=color)
    ax.bar(age_pos, FNR, align='center', color=color)

    labels = ['80-', '60-80', '40-60', '20-40', '0-20', '', 'France', 'China', 'Iran', 'OTHER']


    x_pos = np.arange(len(labels))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
    # ax.set_title(param, fontsize=fontsize)
    ax.set_title('Female', fontsize=fontsize)
    # ax.set_ylim(top=1.0)
    ax.set_ylim([0, 0.07])

    ax.yaxis.grid(True)

    plt.savefig('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results2760/results/FPRFNR/' + param + '.jpg', bbox_inches='tight', dpi=1000)

plotInterFNR_FPR('FNR_F')
# plotInterFNR_FPR('FNR_M')
# plotInterFNR_FPR('FPR_F')
# plotInterFNR_FPR('FPR_M')

