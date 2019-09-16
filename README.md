# cutmix

a Ready-to-use PyTorch Extension of Unofficial CutMix Implementations.

This re-implementation is improved in some parts,

- Fixing [issue #1](https://github.com/clovaai/CutMix-PyTorch/issues/1) in the original repository
- [issue #3](https://github.com/clovaai/CutMix-PyTorch/issues/3) : Random crop regions are randomly chosen, even within the same batch.
- [issue #4](https://github.com/clovaai/CutMix-PyTorch/issues/4) : Different lambda values(sizes of crop regions) are randomly chosen, even within the same batch.
- Images to be cropped are randomly chosen in the whole dataset. Original implementation selects images only inside the same batch(shuffling).
- (TODO) Easy to install and use on your existing project.

Hence, there may be **slightly-improved training results** also.

## Requirements

- python3
- torch >= 1.1.0

## Install

TODO

## Usage

TODO

## Result

### PyramidNet-200 + ShakeDrop + *CutMix* \w CIFAR-100

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

|            |                                 | Top-1 Error(@300epoch) | Top-1 Error(Best) | Model File |
|------------|---------------------------------|------------:|----------:|-----------:|
| ResNet18   | Reported Result \wo CutMix      |             | 30.43     |
|            | Ours                            | 29.674      | 29.56     | [Download](https://www.dropbox.com/s/jdqqbtrwp6mgk7k/model_best.pth.tar?dl=0) |
| ResNet34   | Reported Result \wo CutMix      |             | 26.456    |            |
|            | Ours                            | 24.7        | 24.57     | [Download](https://www.dropbox.com/s/lcjfrcqmuoijig3/model_best.pth.tar?dl=0) |
| ResNet50   | Paper's Reported Result         | N/A         | 21.4      | N/A        |
|            | Author's Code(Our Re-run)       | 21.768      | 21.586    | N/A        |
|            | Our Re-implementation           | 21.524      | 21.340    | [Download(21.25)](https://www.dropbox.com/s/nqell4bh5oj68q1/model_best.pth.tar?dl=0) |

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
