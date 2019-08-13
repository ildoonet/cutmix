# pytorch-cutmix

a Ready-to-use PyTorch Extension of Unofficial CutMix Implementations.

This re-implementation is improved in some parts,

- Fixing [issue #1](https://github.com/clovaai/CutMix-PyTorch/issues/1) in the original repository
- [issue #3] Random crop regions are randomly chosen, even within the same batch.
- [issue #4] lambda values(sizes of crop regions) are randomly chosen, even within the same batch.
- (TODO) Easy to install and use on your existing project.

Hence, there may be slightly-improved training results also.

## Requirements

- python3
- torch >= 1.1.0

## Install

TODO

## Usage

TODO

## Result

### PyramidNet-200 + ShakeDrop + **CutMix** \w CIFAR-100

|                                 | Top-1 Error(@300epoch) | Top-1 Error(Best) | Model File |
|---------------------------------|------------:|------------|------------|
| Paper's Reported Result         | N/A         | 13.81      | N/A        |
| Our Re-implementation           | 13.68       | 13.15      | [Download(12.88)](https://www.dropbox.com/s/q4jsyvvhb4y8ys9/model_best.pth.tar?dl=0)       |

We ran 6 indenpendent experiments with our re-implemented codes and got top-1 errors of 13.09, 13.29, 13.27, 13.24, 13.15 and 12.88, using below command.
(Converged at 300epoch with the top-1 errors of 13.55, 13.66, 13.95, 13.9, 13.8 and 13.32.)


```
$ python train.py -c conf/cifar100_pyramid200.yaml
```

### ResNet + **CutMix** \w ImageNet

TODO

## Reference

- Official
  - Paper : https://arxiv.org/abs/1905.04899
  - Implementation : https://github.com/clovaai/CutMix-PyTorch
- ShakeDrop
  - https://github.com/owruby/shake-drop_pytorch
