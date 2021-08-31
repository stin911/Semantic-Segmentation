import torch

from datetime import datetime


def save_model(net, optimizer, epoch, save_path):
    """
    Save the actual state of the network
    :param optimizer:
    :param net:
    :param save_path:
    :param epoch:
    :return:
    """
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    PATH = str(save_path) + "/" + str(current_time) + "_"+str(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)
