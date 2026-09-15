# CNN Model Training Results

Generated: 2026-09-15 15:04:41
Started: 2026-09-15 14:35:33
Finished: 2026-09-15 15:04:41

## Run Configuration

| Setting | Value |
| --- | --- |
| Dataset | MNIST |
| Image size | 32x32 |
| Classes | 10 |
| Models requested | all |
| Batch size | 128 |
| Epochs requested | 10 |
| Learning rate argument | 0.001 |
| Accelerator | gpu |
| Devices | 1 |
| Num workers | 1 |
| Data directory | ./datasets |

## Model Comparison

| Model | Epochs | Seconds | Parameters | Trainable Parameters | Size (MB) | Optimizer | Learning Rate | Train Loss | Train Acc | Val Loss | Val Acc | Global Step | Log Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | 10 | 152.27 | 103,510 | 103,510 | 0.39 | Adam | 0.001 | 0.0544 | 0.9838 | 0.0958 | 0.9718 | 3,910 | tb_logs/simple_cnn/version_0 |
| lenet5 | 10 | 156.55 | 61,706 | 61,706 | 0.24 | Adam | 0.001 | 0.0340 | 0.9892 | 0.0516 | 0.9859 | 3,910 | tb_logs/lenet5/version_0 |
| alexnet | 10 | 171.91 | 2,364,042 | 2,364,042 | 9.02 | Adam | 0.001 | 0.0391 | 0.9896 | 0.0274 | 0.9916 | 3,910 | tb_logs/alexnet/version_0 |
| googlenet | 10 | 375.71 | 3,571,478 | 3,571,478 | 13.66 | Adam | 0.001 | 0.0205 | 0.9950 | 0.0527 | 0.9867 | 3,910 | tb_logs/googlenet/version_0 |
| vgg16 | 10 | 294.91 | 14,784,394 | 14,784,394 | 56.43 | Adam | 0.001 | 0.0319 | 0.9926 | 0.0481 | 0.9901 | 3,910 | tb_logs/vgg/version_0 |
| resnet50 | 10 | 596.15 | 23,544,970 | 23,544,970 | 90.02 | Adam | 0.001 | 0.0315 | 0.9907 | 0.0711 | 0.9815 | 3,910 | tb_logs/resnet/version_0 |

## Notes

- Metrics are the final values reported by PyTorch Lightning after training.
- Model size includes parameters and buffers stored by the model.
- TensorBoard logs and profiler traces remain in the log directories listed in the table.
