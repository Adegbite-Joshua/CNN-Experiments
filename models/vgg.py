import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torchvision


logger = TensorBoardLogger("tb_logs", name="vgg")

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_vgg"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, skip_first=10, active=20),
)


VGG_CONFIGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def make_vgg_layers(config, in_channels, batch_norm=True):
    layers = []

    for layer in config:
        if layer == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        conv = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(layer), nn.ReLU(inplace=False)])
        else:
            layers.extend([conv, nn.ReLU(inplace=False)])
        in_channels = layer

    return nn.Sequential(*layers)


class VGGModel(L.LightningModule):
    """Building VGG CNN"""

    def __init__(
        self,
        input_size,
        hidden_units,
        num_classes,
        in_channels=1,
        architecture="vgg16",
        batch_norm=True,
    ):
        super().__init__()
        if architecture not in VGG_CONFIGS:
            valid_architectures = ", ".join(VGG_CONFIGS)
            raise ValueError(
                f"Unknown VGG architecture '{architecture}'. "
                f"Choose one of: {valid_architectures}."
            )

        self.input_size = input_size
        self.hidden_units = hidden_units
        self.in_channels = in_channels
        self.architecture = architecture

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.features = make_vgg_layers(
            VGG_CONFIGS[architecture],
            in_channels=in_channels,
            batch_norm=batch_norm,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, hidden_units),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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
            self.logger.experiment.add_image("vgg_images", grid, self.global_step)
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
