import torch
import os
import pandas as pd


def Saved_items(epoches, epoch_losses_train, epoch_losses_val, time_elapsed, batch_size):
    """
    Saves checkpoint of torchvision model during training.
    Args:

        epoch_losses_train: training losses over epochs
        epoch_losses_val: validation losses over epochs

    """
    print('saving')
    time = []
    batch = []
    time.append(time_elapsed)
    batch.append(batch_size)

    df = pd.concat(
        [pd.DataFrame({'epoches': epoches}), pd.DataFrame({'epoch_losses_train': epoch_losses_train}), pd.DataFrame({'epoch_losses_val': epoch_losses_val}),
         pd.DataFrame({'time_elapsed': time}), pd.DataFrame({'batch_size': batch})], axis=1)
    df.to_csv('/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/Saved_items.csv', index=False)
    # state2 = {
    #     'epoch_losses_train': epoch_losses_train,
    #     'epoch_losses_val': epoch_losses_val,
    #     'time_elapsed': time_elapsed,
    #     "batch_size": batch_size
    # }
    # torch.save(state2, r'D:\dataset\CheXpert-v1.0-small\results\Saved_items')
    # torch.save(state2, '/home/user1/data/lyn/CXP/results/Saved_items.csv')


# 保存最后一趟模型
def checkepoch(model, epoch_losses_train, epoch_losses_val, epoches, LR, timelog):
    print('saving each epoch model')
    state = {
        'model': model,
        'epoch_losses_train': epoch_losses_train,
        'epoch_losses_val': epoch_losses_val,
        'epoches': epoches,
        'timelog': timelog,
        'LR': LR
    }
    torch.save(state, '/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/epochModel.tar')


def checkpoint(model, best_loss, best_epoch, LR):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'LR': LR
    }
    # torch.save(state, r'D:\dataset\CheXpert-v1.0-small\results\checkpoint')
    torch.save(state, '/home/qwe/data/cuihaihang/lyn/X_test/4/results_d121/checkpoint.tar')

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 在训练模型的过程中，我们有可能发生梯度爆炸的情况，这样会导致我们模型训练的失败。
# 我们可以采取一个简单的策略来避免梯度的爆炸，那就是梯度截断Clip, 将梯度约束在某一个区间之内，
# 在训练的过程中，在优化器更新之前进行梯度截断操作。
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, LR, dest_dir):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param stn: model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :dest_dir
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer, "learning rate:": LR}
    filename = 'epoch' + str(epoch) + '.pth.tar'
    filename = os.path.join(dest_dir, filename)
    torch.save(state, filename)