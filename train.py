import lightning as L
from dataset import MnistDataModule
# from models.simple_cnn import SimpleCNNModel, logger, profiler
from models.lenet5 import Lenet5Model, logger, profiler
from torchvision import transforms

dm = MnistDataModule(
    data_dir="./datasets",
    batch_size=32,
    num_workers=1,
    transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
)

model = Lenet5Model(
    input_size=1,
    hidden_units=100,
    num_classes=10
)

trainer = L.Trainer(accelerator="cpu", devices=1, min_epochs=2, max_epochs=100, logger=logger, profiler=profiler)

trainer.fit(model, dm)