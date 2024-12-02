import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
import numpy as np
from model_logging import Logger
from wavenet_modules import *


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000):
        self.model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(
            params=self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.logger = logger
        self.logger.trainer = self
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.device = next(model.parameters()).device

    def train(self, batch_size=32, epochs=10, continue_training_at_step=0):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 8 to prevent potential issues
            pin_memory=True
        )
        
        step = continue_training_at_step
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            tic = time.time()
            
            for (x, target) in self.dataloader:
                x = x.to(self.device)
                target = target.view(-1).to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output.squeeze(), target.squeeze())
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                
                self.optimizer.step()
                
                # Get loss value
                loss_value = loss.item()

                # time step duration:
                if step == 100:
                    toc = time.time()
                    print(f"one training step takes approximately {(toc - tic) * 0.01:.3f} seconds")

                if step % self.snapshot_interval == 0 and self.snapshot_path is not None:
                  time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                  save_path = f"{self.snapshot_path}/{self.snapshot_name}_{time_string}"
                  torch.save(self.model.state_dict(), save_path)

                self.logger.log(step, loss_value)
                step += 1

    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        total_samples = 0
        
        with torch.no_grad():
            for (x, target) in self.dataloader:
                x = x.to(self.device)
                target = target.view(-1).to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output.squeeze(), target.squeeze())
                total_loss += loss.item()

                predictions = torch.argmax(output, 1).view(-1)
                correct_pred = torch.eq(target, predictions)
                accurate_classifications += torch.sum(correct_pred).item()
                total_samples += target.numel()

        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = accurate_classifications / total_samples

        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_accuracy


def generate_audio(model, length=8000, temperatures=[0., 1.]):
    """Generate audio samples using the model with different temperature values."""
    model.eval()
    with torch.no_grad():
        samples = []
        for temp in temperatures:
            samples.append(model.generate_fast(length, temperature=temp))
        samples = np.stack(samples, axis=0)
    model.train()
    return samples