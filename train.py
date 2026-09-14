import argparse

import lightning as L
from dataset import MnistDataModule
from torchvision import transforms


IMAGE_SIZE = 32
NUM_CLASSES = 10
MODEL_NAMES = [
    "simple_cnn",
    "lenet5",
    "alexnet",
    "googlenet",
    # "vgg11",
    # "vgg13",
    "vgg16",
    # "vgg19",
    # "resnet18",
    # "resnet34",
    "resnet50",
    # "resnet101",
    # "resnet152",
]


def parse_devices(value):
    if value == "auto":
        return value
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument(
        "--model",
        default="alexnet",
        choices=["all", *MODEL_NAMES],
        help="Model architecture to train. Use 'all' to train every model.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--accelerator",
        default="cpu",
        choices=["cpu", "gpu", "mps", "auto"],
        help="Lightning accelerator to use.",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=1,
        help="Number of devices or 'auto'.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="./datasets")
    return parser.parse_args()


def build_model(model_name, learning_rate):
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNNModel, logger, profiler

        model = SimpleCNNModel(
            input_size=IMAGE_SIZE * IMAGE_SIZE,
            hidden_units=100,
            num_classes=NUM_CLASSES,
        )
    elif model_name == "lenet5":
        from models.lenet5 import Lenet5Model, logger, profiler

        model = Lenet5Model(
            input_size=IMAGE_SIZE * IMAGE_SIZE,
            hidden_units=100,
            num_classes=NUM_CLASSES,
        )
    elif model_name == "alexnet":
        from models.alexnet import AlexNetModel, logger, profiler

        model = AlexNetModel(
            input_size=1,
            hidden_units=100,
            num_classes=NUM_CLASSES,
            in_channels=1,
        )
    elif model_name == "googlenet":
        from models.googlenet import GoogLeNetModel, logger, profiler

        model = GoogLeNetModel(
            input_size=1,
            hidden_units=100,
            num_classes=NUM_CLASSES,
            in_channels=1,
        )
    elif model_name.startswith("vgg"):
        from models.vgg import VGGModel, logger, profiler

        model = VGGModel(
            input_size=1,
            hidden_units=100,
            num_classes=NUM_CLASSES,
            in_channels=1,
            architecture=model_name,
        )
    elif model_name.startswith("resnet"):
        from models.resnet import (
            ResNet18,
            ResNet34,
            ResNet50,
            ResNet101,
            ResNet152,
            logger,
            profiler,
        )

        constructors = {
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
        }
        model = constructors[model_name](
            img_channels=1,
            num_classes=NUM_CLASSES,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, logger, profiler


def main():
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    model_names = MODEL_NAMES if args.model == "all" else [args.model]

    for model_name in model_names:
        print(f"\nTraining {model_name} for {args.epochs} epoch(s)...")
        dm = MnistDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=transform,
        )
        model, logger, profiler = build_model(model_name, args.learning_rate)

        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            min_epochs=1,
            max_epochs=args.epochs,
            logger=logger,
            profiler=profiler,
        )
        trainer.fit(model, dm)


if __name__ == "__main__":
    main()
