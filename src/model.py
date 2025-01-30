import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np


class VoiceModel(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.save_hyperparameters(config)  # save hyperparams, supports OmegaConf

        # number of blocks to ensure the receptive field is as long as the whole sequence
        # assuming a base kernel size == 3
        # dilation is 2**k where k is the block number
        self.num_blocks = int(np.log2(self.config.data.pad_to_length + 2) - 1)

        self.conv_blocks = [
            nn.Conv1d(
                in_channels=1,
                out_channels=self.config.model.conv_channels,  
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",  # make conv output size same as input size
                padding_mode="circular",
            )
        ]

        self.conv_blocks.extend(
            [
                nn.Conv1d(
                    in_channels=self.config.model.conv_channels,
                    out_channels=self.config.model.conv_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=2 ** (k + 1),
                    padding="same",  # make conv output size same as input size
                    padding_mode="circular",
                )
                for k in range(self.num_blocks - 1)
            ]
        )

        self.blocks = [
            ResidualBlock(
                nn.Sequential(
                    conv_block,
                    nn.ReLU(),
                )
            )
            for conv_block in self.conv_blocks
        ]

        self.blocks.extend(
            [
                nn.Conv1d(
                    in_channels=self.config.model.conv_channels,
                    out_channels=1,
                    kernel_size=1,
                ),
                nn.AdaptiveAvgPool1d(1),
            ]
        )

        self.net = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        X, y = batch["waveform"], batch["label"].to(torch.float)
        y_hat = self(X).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["waveform"], batch["label"].to(torch.float)
        y_hat = self(X).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.lr)
        return optimizer


class ResidualBlock(nn.Module):
    # Add a skip connection around a given base_block

    def __init__(self, base_block):
        super().__init__()
        self.base_block = base_block

    def forward(self, x):
        return x + self.base_block(x)
