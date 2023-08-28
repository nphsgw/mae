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
- **Docker Image**: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel 

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

## Modifications

main_pretrain.py
```diff
-assert timm.__version__ == "0.3.2"  # version check
```
```diff
-param_groups = optim_factory.add_weight_decay(model_without_ddp,args.weight_decay)
+param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, weight_decay=args.weight_decay)
```

```diff
-from torch.utils.tensorboard import SummaryWriter
+from torch.utils.tensorboard.writer import SummaryWriter
```

util/misc.py.py
```diff
-from torch._six import inf
-if norm_type == inf:
+if norm_type == torch.inf:
```

models_mae.py
```diff
-Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
+Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)
```

util/pos_embed.py
```diff
-omega = np.arange(embed_dim // 2, dtype=np.float)
+omega = np.arange(embed_dim // 2, dtype=float)
```

## Update exclusively for CIFAR10
main_pretrain.py
```python
transform_train = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0),interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Means and standard deviations for the CIFAR10 data set
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ]
)
```

```python
parser.add_argument("--input_size", default=32, type=int, help="images input size")
parser.add_argument("--data_path", default="./data/CIFAR10", type=str, help="dataset path")
```

## Train

Execute.
```bash
python main_pretrain.py --epochs 100
```

## Visualize

1epoch
![output1](https://github.com/nphsgw/mae/assets/13401073/8c484b78-3c54-4f28-af8a-113b1b995b60)
40epcoh
![output2](https://github.com/nphsgw/mae/assets/13401073/6d8a2ad7-2b38-4ed8-ac72-845e882164fc)
99epoch
![output4](https://github.com/nphsgw/mae/assets/13401073/b9ce165e-d3db-42ce-a20d-b39200d93ade)


For all other details and features, please refer to the original repository's [README](https://github.com/facebookresearch/mae/blob/main/README.md).


