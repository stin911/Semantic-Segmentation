from SemanticSegmentation.engine.TrainingSeg import *
from SemanticSegmentation.DataLoader.CarLoader import *


class Core:
    def __init__(self, args, ):
        self.train = TrainingSeg(args=args)
        self.args = args
        data = DataCarData(args['train_image_path'], args["train_mask_path"],
                           transform=transforms.Compose([ResizeMine(), CToTensor()]))
        self.train_loader = DataLoader(data, batch_size=args['batch_n'], num_workers=args['n_worker'])
        data = DataCarData(args['validation_image_path'], args["validation_mask_path"],
                           transform=transforms.Compose([ResizeMine(), CToTensor()]))
        self.val_loader = DataLoader(data, batch_size=1, num_workers=args['n_worker'])

    def start_train(self):
        self.train.training(self.args["number_epochs"], self.train_loader, self.val_loader)

        save_model(self.train.network, self.train.optimizer, self.args["number_epochs"], self.args['experiment_name'])
