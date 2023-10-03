import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def PlotLearnignCurve():
    print('start to plot')
    LrCurv_param = pd.read_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/Saved_items.csv')
    # LrCurv_param = torch.load(r'D:\paperResources\CheXpert-v1.0-small\results\Saved_items')
    batch_size = LrCurv_param.iloc[0]['batch_size']
    epoch_losses_train =LrCurv_param['epoch_losses_train']
    epoch_losses_val = LrCurv_param['epoch_losses_val']
    epoches = LrCurv_param['epoches']
    print("batch_size", batch_size)
    # print('best_epoch:', best_epoch)

    plt.figure()
    plt.plot(epoches, epoch_losses_train, label="Training Loss")
    plt.plot(epoches, epoch_losses_val, label="Validation Loss")
    my_x_ticks = np.arange(0, len(epoches), 2)
    plt.xticks(my_x_ticks)
    plt.title("Graph of Epoch Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    #   plt.show() Instead of showing the graph, lets save it it as a png file
    # plt.savefig(r'D:\paperResources\CheXpert-v1.0-small\results\epoch_losses.png')
    plt.savefig('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/epoch_losses.png')
