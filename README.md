# pytorch-cutmix

a Ready-to-use PyTorch Extension of Unofficial CutMix Implementations.

This re-implementation is improved in some parts,

- Fixing [issue #1](https://github.com/clovaai/CutMix-PyTorch/issues/1) in the original repository
- [issue #3] Random crop regions are randomly chosen, even within the same batch.
- [issue #4] lambda values(sizes of crop regions) are randomly chosen, even within the same batch.
- (TODO) Easy to install and use on your existing project.

## Requirements

- python3
- torch >= 1.1.0

## Install

TODO

## Usage

TODO

## Result

### PyramidNet-200 + ShakeDrop + *CutMix* \w CIFAR-100

|                                 | Top-1 Error | Model File |
|---------------------------------|------------:|------------|
| Paper's Reported Result         | 13.81       | N/A        |
| Our Re-implementation           | 13.21       | 

We ran 5 indenpendent experiments with our re-implemented codes and got top-1 errors of 13.09, 13.29, 13.27, 13.24, 13.15, using below command.

```
$ python train.py -c conf/cifar100_pyramid200.yaml
```

### ResNet + *CutMix* \w ImageNet

TODO

## Reference

- Official
  - Paper : https://arxiv.org/abs/1905.04899
  - Implementation : https://github.com/clovaai/CutMix-PyTorch
- ShakeDrop
  - https://github.com/owruby/shake-drop_pytorch
