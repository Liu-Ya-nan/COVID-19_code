import time
import csv
import os
# 禁止GPU并行执行，可以更好地提示报错
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"
import datetime
import torch.optim
import torch.utils.data
from torchvision import models
from torch import nn
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
from dataset import Covid
from batchiterator import *
from tqdm import tqdm
import numpy as np


def ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device, LR):

    # Training parameters训练参数
    batch_size = 64

    workers = 2  # mean: how many subprocesses to use for data loading.
    N_LABELS = 2
    start_epoch = 0
    num_epochs = 300  # number of epochs to train for (if early stopping is not triggered)

    epoch_losses_train = []
    epoch_losses_val = []
    epoches = []

    best_loss = 999999
    best_epoch = -1

    timelog = 0

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("Validation_df path", val_df_size)

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df path", train_df_size)

    random_seed = 21    # random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data_retriever = Covid(train_df, path_image=path_image, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
    ]))

    train_loader = torch.utils.data.DataLoader(
        # 从dataset中读取数据，返回：img, label, item["Path"]
        # path_image与train_df的image拼接得到图像路径
        Covid(train_df, path_image=path_image, transform=transforms.Compose([
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(15),
                                                                    transforms.Resize(256),
                                                                      transforms.CenterCrop(256),
                                                                    transforms.ToTensor(),
                                                                    normalize
                                                                ])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print('train_loader为', train_loader)
    valid_data_retriever = Covid(val_df, path_image=path_image, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
    ]))
    val_loader = torch.utils.data.DataLoader(
        Covid(val_df, path_image=path_image, transform=transforms.Compose([
                                                                transforms.Resize(256),
                                                                transforms.CenterCrop(256),
                                                                transforms.ToTensor(),
                                                                normalize
                                                            ])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    path1 = '/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/checkpoint.tar'
    path2 = '/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/epochModel.tar'

    if ModelType == 'densenet169':
        print('没有经过训练，重新构建模型169')
        model = models.densenet169(pretrained=True)
        num_ftrs = model.classifier.in_features
        print('denseNet的特征数量', num_ftrs)  # 1664

        # 在denseNet模型后面加上全连接层nn.Linear（input, output),之后再sigmod
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    if ModelType == 'densenet121':
        print('没有经过训练，重新构建模型121')
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        print('denseNet的特征数量', num_ftrs)               # 1024

        # 在denseNet模型后面加上全连接层nn.Linear（input, output),之后再sigmod
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    if ModelType == 'ResNet50':
        print('没有经过训练，重新构建模型50')
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        print('模型的特征数量', num_ftrs)  # 2048
        # 在denseNet模型后面加上全连接层nn.Linear（input, output),之后再sigmod
        model.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    if ModelType == 'ResNet34':
        print('没有经过训练，重新构建模型34')
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        print('模型的特征数量', num_ftrs)  # 512
        # 在denseNet模型后面加上全连接层nn.Linear（input, output),之后再sigmod
        model.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    if ModelType == 'ResNet18':
        print('没有经过训练，重新构建模型18')
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        print('模型的特征数量', num_ftrs)  # 512
        # 在denseNet模型后面加上全连接层nn.Linear（input, output),之后再sigmod
        model.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    if ModelType == 'Resume':
        print('存在最后一趟epoch的模型')
        lastModel = torch.load(path2)
        model = lastModel['model']
        epoch_losses_train = lastModel['epoch_losses_train']
        epoch_losses_val = lastModel['epoch_losses_val']
        epoches = lastModel['epoches']
        LR = lastModel['LR']
        timelog = lastModel['timelog']
        start_epoch = epoches[-1] + 1

        print('存在最好的模型')
        checkpoint_best = torch.load(path1)
        best_epoch = checkpoint_best['best_epoch']
        best_loss = checkpoint_best['best_loss']
        print('best_epoch', best_epoch)

    # 当有多块GPU时，使用多块GPU加快训练
    if torch.cuda.device_count() > 0:
        print('Using', torch.cuda.device_count(), 'GPUs')
        # model = nn.DataParallel(model)
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        # model = torch.nn.DataParallel(model, device_ids=[1])

    # 将模型加载到GPU或者CPU（device指定使用GPU还是CPU）
    # model = model.to(device)

    if CriterionType == 'BCELoss':
        criterion = nn.BCELoss().to(device)

    since = time.time()
# --------------------------Start of epoch loop
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
# -------------------------- Start of phase

        phase = 'train'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.99))
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        # print('sbsbsbsb')
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train)
        epoches.append(epoch)
        print("Train_losses:", epoch_losses_train)

        phase = 'val'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.99))
        # running_loss, acc_sum = BatchIterator(model=model, phase=phase, Data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device)
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val)
        # accuracy = acc_sum * 100 / len(valid_data_retriever)

        # print("[Epoch " + str(epoch) + " | " + "val loss = " + ("%.7f" % epoch_loss_val) + "  accuracy = " + (
        #         "%.3f" % accuracy) + "%]")

        print("Validation_losses:", epoch_losses_val)


        # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            # 保存训练的的模型
            checkpoint(model, best_loss, best_epoch, LR)

        # log training and validation loss over each epoch
        with open("/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/log_train.log", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if epoch == 0:
                logwriter.writerow(["epoch",  "train_loss", "val_loss", "Seed", "LR"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val, random_seed, LR])

        checkepoch(model, epoch_losses_train, epoch_losses_val, epoches, LR, time.time()-since+timelog)

# -------------------------- End of phase
        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                print("decay loss from " + str(LR) + " to " + str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
                print("created new optimizer with LR " + str(LR))
                # if ((epoch - best_epoch) >= 10):
                #     print("no improvement in 10 epochs, break")
                #     break
        # old_epoch = epoch
    # ------------------------- End of epoch loop
    time_elapsed = time.time() - since + timelog
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    Saved_items(epoches, epoch_losses_train, epoch_losses_val, time_elapsed, batch_size)
    #
    # checkpoint_best = torch.load(r'D:\dataset\CheXpert-v1.0-small\results\checkpoint')
    checkpoint_best = torch.load('/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/checkpoint.tar')
    model = checkpoint_best['model']

    best_epoch = checkpoint_best['best_epoch']
    best_loss = checkpoint_best['best_loss']
    print('best_epoch', best_epoch)
    print('best_loss', best_loss)


    return model, best_epoch


