import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torchvision


logger = TensorBoardLogger("tb_logs", name="alexnet")

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_alexnet"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, skip_first=10, active=20),
)


class AlexNetModel(L.LightningModule):
    """Building AlexNet CNN"""

    def __init__(self, input_size, hidden_units, num_classes, in_channels=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.in_channels = in_channels

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, hidden_units),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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
            self.logger.experiment.add_image("alexnet_images", grid, self.global_step)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
