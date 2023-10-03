import torch
import pandas as pd


# Saving checkpoint.  每一个epoch保存一次模型
def each_checkepoch(i_epoch, model, optimizer):
    print('saving each epoch model')
    path_ckpt = r"/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/each_pth/" + str(i_epoch) + ".pth.tar"
    torch.save({"epoch": i_epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, path_ckpt)
    return path_ckpt


# 保存acc值最高的最好的模型
def bestepoch(i_epoch, model, accuracy, optimizer):
    print('saving the Best epoch model')
    path_ckpt_best = r"/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/best_acc.pth.tar"
    torch.save({"epoch": i_epoch, "model": model, "model_state_dict": model.state_dict(), "accuracy": accuracy,
                "optimizer_state_dict": optimizer.state_dict()}, path_ckpt_best)
    return path_ckpt_best


# 保存最后一趟模型，断点用
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
    torch.save(state, '/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/epochModel.tar')



# 保存损失函数csv文件，画图用
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
    df.to_csv('/home/qwe/data/cuihaihang/lyn/CT3_nobalance/results/Saved_items.csv', index=False)
    # state2 = {
    #     'epoch_losses_train': epoch_losses_train,
    #     'epoch_losses_val': epoch_losses_val,
    #     'time_elapsed': time_elapsed,
    #     "batch_size": batch_size
    # }
    # torch.save(state2, r'D:\dataset\CheXpert-v1.0-small\results\Saved_items')
    # torch.save(state2, '/home/user1/data/lyn/CXP/results/Saved_items.csv')
