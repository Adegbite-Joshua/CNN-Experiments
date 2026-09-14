import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torchvision


logger = TensorBoardLogger("tb_logs", name="googlenet")

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_googlenet"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, skip_first=10, active=20),
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        reduce_3x3,
        out_3x3,
        reduce_5x5,
        out_5x5,
        pool_proj,
    ):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, reduce_3x3, kernel_size=1),
            ConvBlock(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, reduce_5x5, kernel_size=1),
            ConvBlock(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [
                self.branch_1(x),
                self.branch_2(x),
                self.branch_3(x),
                self.branch_4(x),
            ],
            dim=1,
        )


class GoogLeNetModel(L.LightningModule):
    """Building GoogLeNet CNN"""

    def __init__(self, input_size, hidden_units, num_classes, in_channels=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.in_channels = in_channels

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.layers = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(832, hidden_units),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image("googlenet_images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch

        scores = self.forward(x)
        loss = self.loss_fn(scores, y)

        return loss, scores, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
