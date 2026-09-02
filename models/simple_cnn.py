import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torchvision


logger = TensorBoardLogger("tb_logs", name="simple_cnn")
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_simple_cnn"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, skip_first=10, active=20)
)
class SimpleCNNModel(L.LightningModule):
    """Building simple CNN"""

    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
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
            print(f"Shape before: {x.shape}")
            print("\n\n\n\n\n\n\n\n")
            print(f"Shape after: {x.view(-1,1,28,28).shape}")
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
