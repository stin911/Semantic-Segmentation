import torch


def save_model(net, optimizer, epoch, save_path):
    """
    Save the actual state of the network
    :param optimizer:
    :param net:
    :param save_path:
    :param epoch:
    :return:
    """
    PATH = str(save_path) + str(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)
