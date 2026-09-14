import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torchvision


logger = TensorBoardLogger("tb_logs", name="resnet")

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_resnet"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, skip_first=10, active=20),
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
        


class ResNetModel(L.LightningModule):
    """Building ResNet CNN"""

    def __init__(self, block, layers,  image_channels, num_classes, learning_rate=1e-3):
        super().__init__()
        self.in_channels = 64
        self.expansion = block.expansion
        self.learning_rate = learning_rate

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2) # 512 * 4 = 2048 channels at the end
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)       
        
        
    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    stride=stride,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
            
        layers.append(
            block(self.in_channels, out_channels, stride=stride, identity_downsample=identity_downsample)
        )
        self.in_channels = out_channels * self.expansion
        
        for i in range(num_residual_block - 1):
            layers.append(
                block(self.in_channels, out_channels)
            )
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.train_accuracy(scores, y)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.logger is not None and batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image("resnet_images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.val_accuracy(scores, y)
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
        accuracy = self.test_accuracy(scores, y)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def ResNet50(img_channels=3, num_classes=10, learning_rate=1e-3):
    return ResNetModel(block, [3, 4, 6, 3], img_channels, num_classes, learning_rate)


def ResNet18(img_channels=3, num_classes=10, learning_rate=1e-3):
    return ResNetModel(BasicBlock, [2, 2, 2, 2], img_channels, num_classes, learning_rate)


def ResNet34(img_channels=3, num_classes=10, learning_rate=1e-3):
    return ResNetModel(BasicBlock, [3, 4, 6, 3], img_channels, num_classes, learning_rate)


def ResNet101(img_channels=3, num_classes=10, learning_rate=1e-3):
    return ResNetModel(block, [3, 4, 8, 3], img_channels, num_classes, learning_rate)


def ResNet152(img_channels=3, num_classes=10, learning_rate=1e-3):
    return ResNetModel(block, [3, 8, 36, 3], img_channels, num_classes, learning_rate)
