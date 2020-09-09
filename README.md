# cutmix

<img src="https://github.com/clovaai/CutMix-PyTorch/raw/master/img1.PNG" width=50% />

a Ready-to-use PyTorch Extension of Unofficial CutMix Implementations.

This re-implementation is improved in some parts,

- Fixing [issue #1](https://github.com/clovaai/CutMix-PyTorch/issues/1) in the original repository
- [issue #3](https://github.com/clovaai/CutMix-PyTorch/issues/3) : Random crop regions are randomly chosen, even within the same batch.
- [issue #4](https://github.com/clovaai/CutMix-PyTorch/issues/4) : Different lambda values(sizes of crop regions) are randomly chosen, even within the same batch.
- Images to be cropped are randomly chosen in the whole dataset. Original implementation selects images only inside the same batch(shuffling).
- Easy to install and use on your existing project.
- With additional augmentations(fast-autoaugment), the performances are improved further.

Hence, there may be **slightly-improved training results** also.

## Requirements

- python3
- torch >= 1.1.0

## Install

This repository is pip-installable, 

```
$ pip install git+https://github.com/ildoonet/cutmix
```

or you can copy 'cutmix' folder to your project to use it.

## Usage

Our ```CutMix``` is inhereted from the PyTorch Dataset class so you can wrap your own dataset(eg. cifar10, imagenet, ...). Also we provide ```CutMixCrossEntropyLoss```, soft version of cross-entropy loss, which accept soft-labels required by cutmix.

```python
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
...

dataset = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_train)
dataset = CutMix(dataset, num_class=100, beta=1.0, prob=0.5, num_mix=2)    # this is paper's original setting for cifar.
...

criterion = CutMixCrossEntropyLoss(True)
for _ in range(num_epoch):
    for input, target in loader:    # input is cutmixed image's normalized tensor and target is soft-label which made by mixing 2 or more labels.
        output = model(input)
        loss = criterion(output, target)
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Result

### PyramidNet-200 + ShakeDrop + *CutMix* \w CIFAR-100

|                                 | Top-1 Error(@300epoch) | Top-1 Error(Best) | Model File |
|---------------------------------|------------:|------------|------------|
| Paper's Reported Result         | N/A         | 13.81      | N/A        |
| Our Re-implementation           | 13.68       | 13.15      | [Download(12.88)](https://www.dropbox.com/s/q4jsyvvhb4y8ys9/model_best.pth.tar?dl=0)       |
| + Fast AutoAugment              | 13.3        | 12.95      |            |

We ran 6 indenpendent experiments with our re-implemented codes and got top-1 errors of 13.09, 13.29, 13.27, 13.24, 13.15 and 12.88, using below command.
(Converged at 300epoch with the top-1 errors of 13.55, 13.66, 13.95, 13.9, 13.8 and 13.32.)

```bash
$ python train.py -c conf/cifar100_pyramid200.yaml
```

### ResNet + **CutMix** \w ImageNet

|            |                                 | Top-1 Error<br/>(@300epoch) | Top-1 Error<br/>(Best) | Model File |
|------------|---------------------------------|------------:|----------:|-----------:|
| ResNet18   | Reported Result \wo CutMix      | N/A         | 30.43     |
|            | Ours                            | 29.674      | 29.56     | 
| ResNet34   | Reported Result \wo CutMix      | N/A         | 26.456    |            |
|            | Ours                            | 24.7        | 24.57     | [Download](https://www.dropbox.com/s/lcjfrcqmuoijig3/model_best.pth.tar?dl=0) |
| ResNet50   | Paper's Reported Result         | N/A         | 21.4      | N/A        |
|            | Author's Code(Our Re-run)       | 21.768      | 21.586    | N/A        |
|            | Our Re-implementation           | 21.524      | 21.340    | [Download(21.25)](https://www.dropbox.com/s/nqell4bh5oj68q1/model_best.pth.tar?dl=0) |
| ResNet200  | Our Re-implementation           | 
|            | + Fast AutoAugment              | 19.058      | 18.858    | 

```bash
$ python train.py -c conf/imagenet_resnet50.yaml
```

We ran 5 independent experiments on ResNet50. 

- Author's codes
  - 300epoch : 21.762, 21.614, 21.762, 21.644, 21.810
  - best : 21.56, 21.556, 21.666, 21.498, 21.648

- Our Re-implementation
  - 300epoch : 21.53, 21.408, 21.55, 21.4, 21.73
  - best : 21.392, 21.328, 21.386, 21.256, 21.34

## Reference

- Official
  - Paper : https://arxiv.org/abs/1905.04899
  - Implementation : https://github.com/clovaai/CutMix-PyTorch
- ShakeDrop
  - https://github.com/owruby/shake-drop_pytorch
- Fast AutoAugment
  - https://github.com/kakaobrain/fast-autoaugment
