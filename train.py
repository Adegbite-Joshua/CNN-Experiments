import argparse
from datetime import datetime
from pathlib import Path
import time

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
    parser.add_argument(
        "--results-file",
        default="results.md",
        help="Markdown file where the model comparison table will be written.",
    )
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


def count_parameters(model):
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    return total_params, trainable_params


def model_size_mb(model):
    params_size = sum(param.numel() * param.element_size() for param in model.parameters())
    buffers_size = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (params_size + buffers_size) / (1024**2)


def metric_value(metrics, *names):
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        elif hasattr(value, "item"):
            value = value.item()
        return value
    return None


def optimizer_details(trainer):
    if not trainer.optimizers:
        return "-", "-"

    optimizer = trainer.optimizers[0]
    learning_rates = {
        param_group.get("lr")
        for param_group in optimizer.param_groups
        if param_group.get("lr") is not None
    }
    formatted_lrs = ", ".join(str(lr) for lr in sorted(learning_rates))
    return optimizer.__class__.__name__, formatted_lrs or "-"


def format_number(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_int(value):
    return f"{value:,}"


def markdown_table(headers, rows):
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(table)


def write_results_markdown(results_file, args, records, started_at, finished_at):
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Model",
        "Epochs",
        "Seconds",
        "Parameters",
        "Trainable Parameters",
        "Size (MB)",
        "Optimizer",
        "Learning Rate",
        "Train Loss",
        "Train Acc",
        "Val Loss",
        "Val Acc",
        "Global Step",
        "Log Dir",
    ]
    rows = [
        [
            record["model"],
            record["epochs"],
            format_number(record["seconds"], digits=2),
            format_int(record["parameters"]),
            format_int(record["trainable_parameters"]),
            format_number(record["size_mb"], digits=2),
            record["optimizer"],
            record["learning_rate"],
            format_number(record["train_loss"]),
            format_number(record["train_accuracy"]),
            format_number(record["val_loss"]),
            format_number(record["val_accuracy"]),
            format_int(record["global_step"]),
            record["log_dir"],
        ]
        for record in records
    ]

    lines = [
        "# CNN Model Training Results",
        "",
        f"Generated: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finished: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Run Configuration",
        "",
        markdown_table(
            ["Setting", "Value"],
            [
                ["Dataset", "MNIST"],
                ["Image size", f"{IMAGE_SIZE}x{IMAGE_SIZE}"],
                ["Classes", NUM_CLASSES],
                ["Models requested", args.model],
                ["Batch size", args.batch_size],
                ["Epochs requested", args.epochs],
                ["Learning rate argument", args.learning_rate],
                ["Accelerator", args.accelerator],
                ["Devices", args.devices],
                ["Num workers", args.num_workers],
                ["Data directory", args.data_dir],
            ],
        ),
        "",
        "## Model Comparison",
        "",
        markdown_table(headers, rows),
        "",
        "## Notes",
        "",
        "- Metrics are the final values reported by PyTorch Lightning after training.",
        "- Model size includes parameters and buffers stored by the model.",
        "- TensorBoard logs and profiler traces remain in the log directories listed in the table.",
        "",
    ]
    results_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    model_names = MODEL_NAMES if args.model == "all" else [args.model]
    records = []
    started_at = datetime.now()

    for model_name in model_names:
        print(f"\nTraining {model_name} for {args.epochs} epoch(s)...")
        dm = MnistDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=transform,
        )
        model, logger, profiler = build_model(model_name, args.learning_rate)
        total_params, trainable_params = count_parameters(model)
        size_mb = model_size_mb(model)

        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            min_epochs=1,
            max_epochs=args.epochs,
            logger=logger,
            profiler=profiler,
        )
        start_time = time.perf_counter()
        trainer.fit(model, dm)
        seconds = time.perf_counter() - start_time

        metrics = trainer.callback_metrics
        optimizer, actual_learning_rate = optimizer_details(trainer)
        records.append(
            {
                "model": model_name,
                "epochs": trainer.current_epoch,
                "seconds": seconds,
                "parameters": total_params,
                "trainable_parameters": trainable_params,
                "size_mb": size_mb,
                "optimizer": optimizer,
                "learning_rate": actual_learning_rate,
                "train_loss": metric_value(metrics, "train_loss_epoch", "train_loss"),
                "train_accuracy": metric_value(
                    metrics,
                    "train_accuracy_epoch",
                    "train_accuracy",
                ),
                "val_loss": metric_value(metrics, "val_loss"),
                "val_accuracy": metric_value(metrics, "val_accuracy"),
                "global_step": trainer.global_step,
                "log_dir": getattr(logger, "log_dir", "-"),
            }
        )

    finished_at = datetime.now()
    write_results_markdown(args.results_file, args, records, started_at, finished_at)
    print(f"\nSaved results summary to {args.results_file}")


if __name__ == "__main__":
    main()
