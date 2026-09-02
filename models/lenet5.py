import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torchvision


logger = TensorBoardLogger("tb_logs", name="lenet5")

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_lenet5"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, skip_first=10, active=20)
)


class Lenet5Model(L.LightningModule):
    """Building LeNet CNN"""

    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)   
        self.relu = nn.ReLU()     

        self.loss_fn = torch.nn.CrossEntropyLoss()        
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            self.relu,
            self.pool,
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            self.relu,
            self.pool,
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            self.relu,
            nn.Flatten(),
            nn.Linear(120, 84),
            self.relu,
            nn.Linear(84, num_classes),
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
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)
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
