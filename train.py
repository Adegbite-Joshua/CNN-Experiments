import lightning as L
from dataset import MnistDataModule
from models.simple_cnn import SimpleCNNModel, logger

dm = MnistDataModule(
    data_dir="./datasets",
    batch_size=32,
    num_workers=1
)

model = SimpleCNNModel(
    input_size=784,
    hidden_units=100,
    num_classes=10
)

trainer = L.Trainer(accelerator="cpu", devices=1, min_epochs=2, max_epochs=100, logger=logger)

trainer.fit(model, dm)