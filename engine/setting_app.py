import torch
import monai
import monai.networks.nets as nets


def select_loss(name: str):
    if name == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
    elif name == "DICE":
        loss = monai.losses.DiceLoss()
    else:
        raise Exception("The requested loss is not available")
    return loss


def select_optim(name: str, lr, net):
    if name == "ADAM":
        optimizer = torch.optim.Adam(net, lr=lr)
    elif name == "SGD":
        optimizer = torch.optim.SGD(net, lr=lr)
    else:
        raise Exception("The requested loss is not available")
    return optimizer


def select_model(args):
    if args['architecture'] == "basicunet":
        model = nets.basicunet(dimensions=args["dimensions"], in_channels=args["in_channels"],
                               out_channels=args["out_channels"])
    elif args['architecture'] == "else":
        print("")
    else:
        raise Exception("The requested modell is not available")
    return model
