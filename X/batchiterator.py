import torch
from utils import *
import numpy as np
from torch.nn import functional as F


# from evaluation import *


def BatchIterator(model, phase,
                  Data_loader,
                  criterion,
                  optimizer,
                  device):
    # --------------------  Initial paprameterd
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 2000
    running_loss = 0.0

    outs = []
    gts = []
    acc_sum = []

    for i, data in enumerate(Data_loader):
        imgs, labels, _ = data
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:
            for label in labels.cpu().numpy().tolist():
                gts.append(label)
            # target = torch.max(labels)
            # labels_y = target.cpu().numpy().tolist()

            # 不启用BatchNormalization和Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，
            model.eval()
            with torch.no_grad():
                outputs = model(imgs)
                # prediction = torch.max(outputs)
                # pred_y = prediction.cpu().numpy().tolist()

            # out = torch.sigmoid(outputs).data.cpu().numpy()
            # outs.extend(out)
            # outs = np.array(outs)
            # gts = np.array(gts)
        # evaluation_items(gts, outs)

        # #  修改了一下：target nelement (528) != input nelement (672)
        # outputs = F.sigmoid(outputs)
        # outputs = outputs.view(outputs.size(0), -1)
        # labels = labels.view(labels.size(0), -1)
        # print(outputs)
        # print(labels)
        # # 修改结束

        loss = criterion(outputs, labels)

        if phase == 'train':
            loss.backward()  # 回传损失，过程中会计算梯度
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights

        running_loss += loss.item() * batch_size

        # if phase == 'val':
        #     acc_sum += sum((pred_y - labels_y) == 0)

        if i % 1000 == 0:  # 每1个batch输出一次
            print(str(i * batch_size))
            # print('第', i * batch_size, '个batch')

    # if phase == 'train':
    return running_loss
    # if phase == 'val':
    #     return running_loss, acc_sum
