# Code for "Efficient Deep Demosaicing with Spatially Downsampled Isotropic Networks" @ WACVW 2026.

[Paper](https://arxiv.org/abs/2601.00703)

## Usage for Hard Demosaicing Dataset

Download the data from [here](https://github.com/SamsungLabs/unified-demosaicing).

Change the options in unidem/configs/*/*.yaml (should reflect the script you are trying to use).

From within unidem directory, run with:
python PyTorch/runner.py --config *insert your config file here*. 

## Usage for HybridEVS/Classic Bayer Demosaicing

Training: 

Download the ImageNet data and change the yml file in demosaic/options to reflect the file path.

For validation, download the Div2K validation set, and change the yml file similarly. 

From within demosaic directory, train with:
python basicsr/train.py -opt *insert your config file here*

For evaluation, download a dataset (Kodak24 or etc), and then create a demosaic/dataset directory, and place the dataset there. 

From within demosaic directory, test with:
python basicsr/test.py -opt *insert your config file here*
# jd3net
