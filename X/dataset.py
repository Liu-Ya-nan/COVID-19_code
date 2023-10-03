import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import imageio
# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent.parent

class Covid(Dataset):
    # init初始化函数，为整个class提供一个全局变量
    def __init__(self, dataframe, path_image, finding="any", transform=None):
        self.dataframe = dataframe
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform
        self.path_image = path_image

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation

            # 如果dataframe（csv)中有列名"any"
            if finding in self.dataframe.columns:

                if len(self.dataframe[self.dataframe[finding] == 1]) > 0:
                    self.dataframe = self.dataframe[self.dataframe[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.PRED_LABEL = [
            'COVID-19',
            'Normal']

    def __getitem__(self, idx):
        # 读取csv某一行数据
        item = self.dataframe.iloc[idx]
        # 得到图片的绝对路径--拼接而成
        img = imageio.imread(os.path.join(self.path_image, item["Pid"], item["Path"]), pilmode='RGB')
        # img = imageio.imread(BASE_DIR.joinpath(self.path_image, item["Pid"], item["Path"]), pilmode='RGB')
        # 将数组转化为图片
        img = Image.fromarray(img)

        # 图像增强处理
        if self.transform is not None:
            img = self.transform(img)
            # print(img)     # 输出为tensor()

        # label = np.zeros(len(self.PRED_LABEL), dtype=int)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            # strip()去除首尾空格
            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')
            # print(label)
        return img, label, item["Path"]  # self.dataframe.index[idx]

    def __len__(self):
        return self.dataset_size
