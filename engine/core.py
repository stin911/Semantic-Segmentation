from SemanticSegmentation.engine.TrainingSeg import *
from SemanticSegmentation.DataLoader.CarLoader import *
import os


class Core:
    def __init__(self, args, ):
        self.train = TrainingSeg(args=args, train=args["train"])
        self.args = args
        self.train_loader = None
        self.val_loader = None
        self.load_data()

    def load_data(self):

        # load the data for training and validation
        # in this case both the image and the mask are available
        if self.args["train"]:
            data = DataCarData(self.args['train_image_path'], self.args["train_mask_path"],
                               transform=transforms.Compose([ResizeMine(), CToTensor()]))
            self.train_loader = DataLoader(data, batch_size=self.args['batch_n'], num_workers=self.args['n_worker'])

            data = DataCarData(self.args['validation_image_path'], self.args["validation_mask_path"],
                               transform=transforms.Compose([ResizeMine(), CToTensor()]))
            self.val_loader = DataLoader(data, batch_size=1, num_workers=self.args['n_worker'])

        # load the data for inference
        # in this scenario we don't have the mask
        else:
            data = DataCarData(self.args['train_image_path'], self.args["train_mask_path"],
                               transform=transforms.Compose([ResizeMine(), CToTensor()]))
            self.train_loader = DataLoader(data, batch_size=1, num_workers=self.args['n_worker'])

    def start_train(self):
        self.train.loading(os.path.abspath(self.args["load"]))
        self.train.training(self.args["number_epochs"], self.train_loader, self.val_loader)

        save_model(self.train.network, self.train.optimizer, self.args["number_epochs"],
                   os.path.abspath(self.args['experiment_name']))

    def inference(self):
        if self.args["load"] is not None:
            self.train.loading(os.path.abspath(self.args["load"]))
        self.train.inference(data=self.train_loader)
