from SemanticSegmentation.engine.TrainingSeg import *
from SemanticSegmentation.DataLoader.CarLoader import *
from torchvision.transforms import transforms
from SemanticSegmentation.general_utils.general_transform import *


class inference:
    def __init__(self, args):
        self.train = TrainingSeg(args=args, train=False)
        self.args = args
        data = DataCarData(args['path_image'], None,
                           transform=transforms.Compose([ResizeMine(), CToTensor()]))
        self.data = DataLoader(data, batch_size=1, num_workers=args['n_worker'])

    def predict(self):
        self.train.loading(os.path.abspath(self.args["load"]))
        self.train.inference(data=self.data)
