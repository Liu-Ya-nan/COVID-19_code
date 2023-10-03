import torch
import torch.nn as nn
from torch.nn import functional as F
from FocalLoss import FocalLoss

def BatchIterator(model, phase,
                  Data_loader,
                  optimizer,
                  device, loss_sum, acc_sum):
    # num_rev_0, num_rev_1 = 1 / 21236, 1 / 256185
    num_rev_0, num_rev_1 = 0.1, 0.9
    # num_rev_0 = num_rev_0.to(device)
    # num_rev_1 = num_rev_1.to(device)
    # weights = torch.tensor([num_rev_0, num_rev_1]).to(device)
    weights = torch.tensor([num_rev_1]).to(device)
    criterion = FocalLoss().to(device)

    if phase == "train":
        # for step, (data, label) in enumerate(Data_loader):
        #     img = data.to(device)
        #     print('label是啥222', label)   # tensor([[1., 0.]])
        #     # print(img.shape)
        #     targets = label.to(device)
        #     # print('targets是啥！！', targets)    # tensor([[1., 0.]], device='cuda:0')
        #     outputs = model(img).squeeze(1)
        #     print('第3次！！！', torch.max(targets, 1)[1])
        for step, imgs in enumerate(Data_loader):
            data, label = imgs
            img = data.to(device)
            # print(img.shape)
            targets = label.to(device)
            # print('targets是啥！！', targets)    # tensor([[1., 0.]], device='cuda:0')
            outputs = model(img).squeeze(1)
            # 按照（0，1）0比较少，权重大，loss结果应该变小才对
            # loss = F.cross_entropy(outputs, torch.max(targets, 1)[1]).to(device)
            loss = criterion(outputs, torch.max(targets, 1)[1]).to(device)

            # loss = F.cross_entropy(outputs, torch.max(targets, 1)[1], weight=weights, size_average=True).to(device)
            loss_sum += loss.detach().item()
            # print('run ', step, loss_sum)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_sum
    if phase == "val":
        # for step, (data, label
        #
        # ) in enumerate(Data_loader):
        #     img = data.to(device)
        #     # print(img.shape)
        #     print('label是啥val11', label)   # tensor([[1., 0.]])
        #
        #     targets = label.to(device)
        #
        #     outputs = model(img).squeeze(1)
        #     print('label是啥val22', torch.max(targets, 1)[1])   # tensor([[1., 0.]])
        for step, imgs in enumerate(Data_loader):
            data, label = imgs
            img = data.to(device)
            # print(img.shape)

            targets = label.to(device)

            outputs = model(img).squeeze(1)

            target = torch.max(targets, 1)[1]
            target_y = target.data.cpu().numpy()

            # loss = F.cross_entropy(outputs, target).to(device)
            loss = criterion(outputs, target).to(device)

            # loss = F.cross_entropy(outputs, target, weight=weights, size_average=True).to(device)
            loss_sum += loss.detach().item()

            prediction = torch.max(outputs, 1)[1]
            pred_y = prediction.data.cpu().numpy()

            acc_sum += sum((pred_y - target_y) == 0)
        return loss_sum, acc_sum


# if __name__=='__main__':
