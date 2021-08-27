import torch
from abc import ABC, abstractmethod
from SemanticSegmentation.AbstractClass.Training import Training
from SemanticSegmentation.general_utils.Utilis_model import *


class TrainingSeg(Training):

    def training(self, epochs, data, val):
        for epoch in range(epochs):
            running_loss = 0

            for batch_n, sample in enumerate(data):
                self.optimizer.zero_grad()
                image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)

                output = self.network(image)
                mask = mask.unsqueeze(1)

                loss = self.criterion(output, mask)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            t_loss = running_loss / (len(data) / 4)
            v_loss = self.inference(val)
            self.writer.add_scalar("Loss/train", t_loss, epoch)
            self.writer.add_scalar("Loss/val", v_loss, epoch)
            save_model(self.network, self.optimizer, epoch, )
            return t_loss, v_loss

    def inference(self, data):
        running_loss = 0
        for batch_n, sample in enumerate(data):
            image, mask = sample["image"].to(self.device), sample['mask'].to(self.device)

            with torch.no_grad():
                output = self.network(image)
                mask = mask.unsqueeze(1)
                loss = self.criterion(output, mask)
                running_loss += loss.item()

        f_loss = running_loss / len(data)
        return f_loss
