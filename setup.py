from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.1'

# 'opencv-python >= 3.3.1'
REQUIRED_PACKAGES = [
]

DEPENDENCY_LINKS = [
]

setuptools.setup(
    name='cutmix',
    version=_VERSION,
    description='a Ready-to-use PyTorch Extension of Unofficial CutMix Implementations',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/ildoonet/cutmix/',
    license='MIT License',
    package_dir={},
    packages=setuptools.find_packages(exclude=['run', 'autoaug', 'conf', 'network', 'tests']),
)
