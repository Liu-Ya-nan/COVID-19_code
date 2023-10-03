import time
import csv
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import tqdm
from tqdm import *
from Dataset import GetLoader
from model import generate_model
from batchiterator import *
from utils import *


def _init_fn(worker_id):
    np.random.seed(int(12) + worker_id)


def validation(valid_loader, path_ckpt, ModelType, sample_size, sample_duration, optimizer, device):

    if ModelType == 'ResNet18':
        model = generate_model(model_type='resnet', model_depth=18,
                               sample_size=sample_size,
                               sample_duration=sample_duration)

        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        # model = torch.nn.DataParallel(model).to(device)
        model.eval()
        model.to(device)

    if ModelType == 'ResNet34':
        model = generate_model(model_type='resnet', model_depth=34,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'ResNet50':
        model = generate_model(model_type='resnet', model_depth=50,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'densenet101':
        model = generate_model(model_type='densenet', model_depth=101)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'densenet121':
        model = generate_model(model_type='densenet', model_depth=121)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'mobilenet':
        model = generate_model(model_type='mobilenet',
                               sample_size=sample_size)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'mobilenetv2':
        model = generate_model(model_type='mobilenetv2',
                               sample_size=sample_size)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'c3d':
        model = generate_model(model_type='c3d',
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'squeezenet':
        model = generate_model(model_type='squeezenet',
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'shufflenet':
        model = generate_model(model_type='shufflenet')
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'shufflenetv2':
        model = generate_model(model_type='shufflenetv2',
                               sample_size=sample_size)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'resnext50':
        model = generate_model(model_type='resnext', model_depth=50,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'resnext101':
        model = generate_model(model_type='resnext', model_depth=101,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'resnext152':
        model = generate_model(model_type='resnext', model_depth=152,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

    if ModelType == 'Resume':
        print('存在最后一趟epoch的模型')
        lastModel = torch.load('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/epochModel.tar')
        model = lastModel['model']
        epoch_losses_train = lastModel['epoch_losses_train']
        epoch_losses_val = lastModel['epoch_losses_val']
        epoches = lastModel['epoches']
        LR = lastModel['LR']
        timelog = lastModel['timelog']
        start_epoch = epoches[-1] + 1

        print('存在acc最好的模型')
        checkpoint_best = torch.load('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/best_acc.pth.tar')
        best_epoch = checkpoint_best['epoch']
        best_acc = checkpoint_best['accuracy']
        print('best_epoch', best_epoch)
        print('best_acc', best_acc)

    loss_sum = 0
    acc_sum = 0
    phase = 'val'
    loss_sum, acc_sum = BatchIterator(model=model, phase=phase, Data_loader=valid_loader, optimizer=optimizer,
                                      device=device,
                                      loss_sum=loss_sum, acc_sum=acc_sum)
    return loss_sum, acc_sum


def train(ModelType, data_train, data_val, sample_size, sample_duration, device, LR, MAX_EPOCH,
          batch_size):
    MAX_EPOCH = MAX_EPOCH
    batch_size = batch_size
    random_seed = 21  # random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    timelog = 0
    start_epoch = 1

    epoches = []
    epoch_losses_train = []
    epoch_losses_val = []
    best_loss = 999999

    data_train = data_train
    data_val = data_val
    train_data_retriever = GetLoader(data_train)
    valid_data_retriever = GetLoader(data_val)

    train_loader = torch_data.DataLoader(train_data_retriever, batch_size=batch_size, shuffle=True, num_workers=4,
                                         pin_memory=False, worker_init_fn=_init_fn)
    valid_loader = torch_data.DataLoader(valid_data_retriever, batch_size=batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False, worker_init_fn=_init_fn)

    if ModelType == 'DenseNetModel':
        model = generate_model(model_type='DenseNetModel', model_depth=18,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        # model = torch.nn.DataParallel(model).to(device)
        model.train()
        model.to(device)

    if ModelType == 'ResNet18':
        model = generate_model(model_type='resnet', model_depth=18,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        # model = torch.nn.DataParallel(model).to(device)
        model.train()
        model.to(device)

    if ModelType == 'ResNet34':
        model = generate_model(model_type='resnet', model_depth=34,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'ResNet50':
        model = generate_model(model_type='resnet', model_depth=50,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'densenet101':
        model = generate_model(model_type='densenet', model_depth=101)
        model.train()
        model.to(device)

    if ModelType == 'densenet121':
        model = generate_model(model_type='densenet', model_depth=121)
        model.train()
        model.to(device)

    if ModelType == 'mobilenet':
        model = generate_model(model_type='mobilenet',
                               sample_size=sample_size)
        model.train()
        model.to(device)

    if ModelType == 'mobilenetv2':
        model = generate_model(model_type='mobilenetv2',
                               sample_size=sample_size)
        model.train()
        model.to(device)

    if ModelType == 'c3d':
        model = generate_model(model_type='c3d',
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'squeezenet':
        model = generate_model(model_type='squeezenet',
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'shufflenet':
        model = generate_model(model_type='shufflenet')
        model.train()
        model.to(device)

    if ModelType == 'shufflenetv2':
        model = generate_model(model_type='shufflenetv2',
                               sample_size=sample_size)
        model.train()
        model.to(device)

    if ModelType == 'resnext50':
        model = generate_model(model_type='resnext', model_depth=50,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'resnext101':
        model = generate_model(model_type='resnext', model_depth=101,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'resnext152':
        model = generate_model(model_type='resnext', model_depth=152,
                               sample_size=sample_size,
                               sample_duration=sample_duration)
        model.train()
        model.to(device)

    if ModelType == 'Resume':
        print('存在最后一趟epoch的模型')
        lastModel = torch.load('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/epochModel.tar')
        model = lastModel['model']
        epoch_losses_train = lastModel['epoch_losses_train']
        epoch_losses_val = lastModel['epoch_losses_val']
        epoches = lastModel['epoches']
        LR = lastModel['LR']
        timelog = lastModel['timelog']
        start_epoch = epoches[-1] + 1

        print('存在acc最好的模型')
        checkpoint_best = torch.load('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/best_acc.pth.tar')
        best_epoch = checkpoint_best['epoch']
        best_acc = checkpoint_best['accuracy']
        print('best_epoch', best_epoch)
        print('best_acc', best_acc)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_valid_score = 0
    # writer = SummaryWriter(comment='Linear')

    since = time.time()
    for i_epoch in tqdm(range(start_epoch, MAX_EPOCH + 1)):

        print('Epoch {}/{}'.format(i_epoch, MAX_EPOCH))
        print('-' * 20)
        loss_sum = 0
        acc_sum = 0
        # ------train--------------------------------------------------------------------------------
        phase = 'train'
        loss_sum = BatchIterator(model=model, phase=phase, Data_loader=train_loader, optimizer=optimizer, device=device,
                                 loss_sum=loss_sum, acc_sum=acc_sum)
        loss_avg = loss_sum / len(train_loader)
        epoch_losses_train.append(loss_avg)
        epoches.append(i_epoch)
        print("[Epoch " + str(i_epoch) + " | " + "train loss = " + ("%.7f" % loss_avg) + "]")
        print("Train_losses:", epoch_losses_train)

        # Saving checkpoint.每一个epoch保存一次模型
        path_ckpt = each_checkepoch(i_epoch, model, optimizer)
        # ------val-------------------------------------------------------------------------------
        loss_sum, acc_sum = validation(valid_loader, path_ckpt, ModelType, sample_size,
                                       sample_duration, optimizer, device=device)
        loss_val = loss_sum / len(valid_loader)
        # acc指的是，预测对的占总数据量的百分比
        accuracy = acc_sum * 100 / len(valid_data_retriever)
        epoch_losses_val.append(loss_val)

        print("[Epoch " + str(i_epoch) + " | " + "val loss = " + ("%.7f" % loss_val) + "  accuracy = " + (
                "%.3f" % accuracy) + "%]")
        print("Validation_losses:", epoch_losses_val)

        # 保存acc值最高的最好的模型
        if best_valid_score < accuracy:
            best_valid_score = accuracy
            path_ckpt_best = bestepoch(i_epoch, model, accuracy, optimizer)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = i_epoch

        # log training and validation loss over each epoch
        with open("/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/log_train.log", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if i_epoch == 0:
                logwriter.writerow(["epoch", "train_loss", "val_loss", "Seed", "LR"])
            logwriter.writerow([i_epoch, loss_avg, loss_val, random_seed, LR])

        # 保存最后一趟模型
        checkepoch(model, epoch_losses_train, epoch_losses_val, epoches, LR, time.time() - since + timelog)

        if ((i_epoch - best_epoch) >= 3):
            if loss_val > best_loss:
                print("decay loss from " + str(LR) + " to " + str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
                print("created new optimizer with LR " + str(LR))

    time_elapsed = time.time() - since + timelog
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 保存损失函数csv文件，画图用
    Saved_items(epoches, epoch_losses_train, epoch_losses_val, time_elapsed, batch_size)

    # 打开acc最高的模型，查看其参数
    checkpoint_best = torch.load('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/best_acc.pth.tar')
    model = checkpoint_best['model']
    best_epoch = checkpoint_best['epoch']
    best_acc = checkpoint_best['accuracy']
    print('best_epoch', best_epoch)
    print('best_acc', best_acc)

    return model, best_epoch

# if __name__ == '__main__':
#     train()
