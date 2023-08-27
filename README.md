# Changes from the Original Repository

This repository is a fork of [MAE](https://github.com/facebookresearch/mae). 

## Purpose of Changes

The primary objectives behind the changes made in this fork are:

- **Objective 1**: To operate in the latest PyTorch environment.
- **Objective 2**: To ensure functionality with RTX A4500.

## Tested Environments

This fork has been tested and confirmed to work in the following environments:

- **OS**: Ubuntu22.04
- **GPU**: RTX A4500


```python
import torch,torchvision,timm

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
print(timm.__version__)

#2.0.1
#0.15.2
#True
#0.9.5

```

```bash
$ nvidia-smi
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A4500                On | 00000000:01:00.0 Off |                  Off |
| 30%   40C    P8               28W / 200W|   1597MiB / 20470MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+

$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0

```


For all other details and features, please refer to the original repository's [README](https://github.com/facebookresearch/mae/blob/main/README.md).

