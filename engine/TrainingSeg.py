import torch
from abc import ABC, abstractmethod
from SemanticSegmentation.AbstractClass.Training import Training
from SemanticSegmentation.general_utils.Utilis_model import *
from SemanticSegmentation.general_utils.logging import *
import numpy as np
from SemanticSegmentation.general_utils.Utils_visual import *


class TrainingSeg(Training):

    def training(self, epochs, data, val):
        for epoch in range(epochs):

            running_loss = 0
            self.network.train()
            for batch_n, sample in enumerate(data):
                self.optimizer.zero_grad()
                image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)

                output = self.network(image)
                mask = mask.unsqueeze(1)

                loss = self.criterion(output, mask)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            t_loss = running_loss / (len(data) / self.args['batch_n'])
            v_loss = self.validation(val)
            self.writer.add_scalar("Loss/train", t_loss, epoch)
            self.writer.add_scalar("Loss/val", v_loss, epoch)
            log_funct(epoch=epoch, training_l=t_loss, validation_l=v_loss)

    def validation(self, data):
        running_loss = 0
        self.network.eval()
        for batch_n, sample in enumerate(data):
            image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)

            with torch.no_grad():
                output = self.network(image)
                mask = mask.unsqueeze(1)
                loss = self.criterion(output, mask)
                running_loss += loss.item()

        f_loss = running_loss / len(data)
        return f_loss

    def inference(self, data):
        self.network.eval()
        for i_batch, sample in enumerate(data):
            if i_batch < 10:
                image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)
                with torch.no_grad():
                    output = self.network(image)
                    original = image.detach().cpu().numpy()[0]
                    res = output.detach().cpu().numpy()[0]
                    res = np.where(res >= 0.5, 1, 0).astype(int)
                    f = res * (original * 255).astype(int)
                    plot(f)
            else:
                break

    def loading(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
