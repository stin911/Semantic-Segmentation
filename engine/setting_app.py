import torch
import monai
import monai.networks.nets as nets


class Combined:
    def __init__(self, weight=None):
        self.weight = weight

    def __call__(self, output, target):
        self.forward(output, target)

    def forward(self, output, target):

        return CombinedBCEDICE(output, target,self.weight)


def CombinedBCEDICE(output, target,w):
    a = torch.nn.BCEWithLogitsLoss()(output, target)
    b = monai.losses.DiceLoss()(output, target)
    loss = a + b
    return loss


def select_loss(name: str):
    if name == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
    elif name == "DICE":
        loss = monai.losses.DiceLoss()
    elif name == "combine":
        loss = Combined()
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
