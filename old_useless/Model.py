import monai.networks.nets as nets
from SemanticSegmentation.DataLoader.CarLoader import *
from torch.utils.data import DataLoader
from torchsummary import summary
from SemanticSegmentation.general_utils.general_transform import *
from SemanticSegmentation.general_utils.Utils_visual import *
from torch.utils.tensorboard import SummaryWriter
from SemanticSegmentation.general_utils.Utils_data import save_model


class model:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nets.basicunet(dimensions=args["dimensions"], in_channels=args["in_channels"], out_channels=1) \
            .to(self.device)


class Training_seg:
    def __init__(self, batch, num_work, args):
        mod = model(args=args)
        self.net = mod.net
        self.batch_n = batch
        self.num_work = num_work
        self.train_loader = None
        self.val_loader = None
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0001, momentum=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter()

    def initialize_data(self, t_image_path, t_mask_path, v_image_path=None, v_mask_path=None):
        data = DataCarData(t_image_path, t_mask_path, transform=transforms.Compose([ResizeMine(), CToTensor()]))

        self.train_loader = DataLoader(data, batch_size=self.batch_n, num_workers=self.num_work)
        data = DataCarData(v_image_path, v_mask_path, transform=transforms.Compose([ResizeMine(), CToTensor()]))
        self.val_loader = DataLoader(data, batch_size=1, num_workers=self.num_work)

    def start1(self):

        for i, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)

            output = self.net(image)
            mask: torch.Tensor = mask.to(torch.long)
            mask = mask.unsqueeze(1)

            loss = self.criterion(output, mask)
            print(loss.item())

            # plot(out.cpu().detach().numpy()[0])

    def start(self, n_epoc):
        for i in range(6, n_epoc):
            print("Epoch: " + str(i))
            t_loss = self.training(self.train_loader)
            v_loss = self.training(self.val_loader, False)
            self.writer.add_scalar("Loss/train", t_loss, i)
            self.writer.add_scalar("Loss/val", v_loss, i)

            save_model(self.net, self.optimizer, i, "D:/Car-dataset/ModelSave/")

    def training(self, data, TRAIN=True):
        running_loss = 0
        if TRAIN:
            self.net.train()
            for i_batch, sample in enumerate(data):
                print(i_batch)
                self.optimizer.zero_grad()
                image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)
                output = self.net(image)
                # mask: torch.Tensor = mask.to(torch.long)
                mask = mask.unsqueeze(1)
                loss = self.criterion(output, mask)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            f_loss = running_loss / (len(data) / 4)

        else:
            self.net.eval()
            for i_batch, sample in enumerate(data):
                image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)
                with torch.no_grad():
                    output = self.net(image)
                    # mask: torch.Tensor = mask.to(torch.long)
                    mask = mask.unsqueeze(1)
                    loss = self.criterion(output, mask)
                    running_loss += loss.item()
            f_loss = running_loss / len(data)

        return f_loss

    def summary(self):
        """
        :return: generate a summary of the model
        """
        summary(self.net, (3, 512, 512))

    def load_inf(self):
        checkpoint = torch.load("D:/Car-dataset/ModelSave/6.pth")
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def show_result(self):
        self.net.eval()
        for i_batch, sample in enumerate(self.train_loader):
            image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)
            with torch.no_grad():
                output = self.net(image)
                res = output.detach().cpu().numpy()[0]
                res = np.where(res >= 0.5, 1, 0)
                print(res)
                plot(res)

class Training:
    def __init__(self, args):
        self.network = args['model']
        self.criterion = args['loss']
        self.optimizer = args['optimizer']
        self.data_loader = args['training']
        self.val_loader = args['validation']


def main():
    sett = {'dimensions': 2, 'in_channels': 3}
    training = Training_seg(batch=4, num_work=4, args=sett)
    training.initialize_data("D:/Car-dataset/train/", "D:/Car-dataset/Mask_train/", "D:/Car-dataset/val_image/",
                             "D:/Car-dataset/val_mask/")
    training.start(50)


if __name__ == '__main__':
    main()
