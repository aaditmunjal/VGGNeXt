# A VGGNet for the 2020s: Modernizing a Classic ConvNet
By Aadit Munjal

## Introduction

In 2022, Liu et al. introduced ConvNeXt, a pure ConvNet that challenges the dominance of vision transformers. The authors modernized a standard ResNet by gradually incorporating design decisions from Transformers, ultimately creating an architecture that exceeded the performance of the state-of-the-art vision transformers at the time. The primary goal of this study is to validate the generalizability of Liu et al.'s work, and develop a modern VGG architecture fit for the 2020s. We start with the VGG16 architecture and systematically modernize it towards the design of a hierarchical vision transformer. The culmination of these modernization efforts is VGGNeXt, a modern VGG architecture that outperforms the original architecture for significantly lesser computation. Overall, our results validate the principles presented in ConvNeXt, and motivate further research into modernizing newer ConvNets to compete with the current state-of-the-art vision transformers.

## Model Performance

| Iteration | Method                   | Validation Accuracy (%) | Compute (GFLOPs) | Params (M) |
|:----------|:-------------------------|-----------------------:|------------------:|-----------:|
| 0         | VGG16                    | 50.42                   | 1.38             | 135.09     |
| 0.5       | VGG16 128x128            | 56.51                   | 5.15             | 135.09     |
| 1         | Stage Ratio              | 57.59                   | 5.60             | 133.32     |
| 2         | Four Stage               | 55.27                   | 1.72             | 129.78     |
| 3         | GELU                     | 56.18                   | 1.72             | 129.78     |
| 4         | Separate Downsampling    | 52.84                   | 2.21             | 135.24     |
| 4.5       | Hybrid Downsampling      | 53.15                   | 1.91             | 134.40     |
| 5         | Block                    | 54.10                   | 1.87             | 132.14     |
| 6         | Residual Block           | 52.54                   | 2.36             | 137.60     |
| 7         | Residual Block V2        | 49.58                   | 0.71             | 135.27     |
| 8         | VGGNeXt                  | 51.66                   | 1.12             | 138.44     |

## Dataset

This project utilizes the Tiny ImageNet dataset, featuring 100000 training images (500 per class) and 10000 validation images (50 per class) belonging to 200 classes. Each image is a 64x64 RGB image derived from the larger ImageNet-1k dataset


## Project Structure

```
VGGNeXt/
├── vggnext.py                     # The final VGGNeXt architecture
├── vgg.py                         # All iterations starting with the base VGG16
├── main.py                        # Main file for training the model
├── tune.py                        # AdamW hyperparameter tuning
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── plots/                         # Validation accuracy over epochs for each model iteration
├── adamw/                         # Validation accuracy over epochs with the adamW optimizer
```

## Setup & Installation


1.  Clone the repository:
    ```bash
    git clone https://github.com/aaditmunjal/VGGNeXt.git
    cd VGGNeXt
    ```
2.  Create and activate virtual environment (highly recommended)

3.  To install all necessary packages, the following command can be utilized:
    ```bash
    pip install -r requirements.txt
    ```

    To ensure full reproducibility of the results, the exact package versions used in the experiments are listed in ```requirements.txt```. 
    If these versions are not compatible with your system, you may need to install or downgrade to versions that are supported by your machine.

4.  Finally, to train the model use the following command:
    ```bash
    python3 main.py
    ```

    By default, ```main.py``` trains the final VGGNeXt architecture. However, the code has been set up to allow running any model iteration. Simply swap ```vggnext``` with the appropriate model from ```vgg.py```. To accurately reproduce the results for any given iteration, the corresponding training regimen must be followed. The next section outlines the training configuration for each iteration for ease of reference.


## Training Setup

| Iteration | Method                  | Optimizer | Image Size | Epochs | Cosine Annealing | Warmup     |
|:----------|:------------------------|:----------|:-----------|:-------|:-----------------|:-----------|
| 0         | VGG16                   | SGD       | 64x64      | 100    | No               | -          |
| 0.5       | VGG16 128x128           | SGD       | 128x128    | 100    | No               | -          |
| 1         | Stage Ratio             | SGD       | 128x128    | 100    | No               | -          |
| 2         | Four Stage              | SGD       | 64x64      | 100    | No               | -          |
| 3         | GELU                    | SGD       | 64x64      | 100    | No               | -          |
| 4         | Separate Downsampling   | SGD       | 64x64      | 100    | No               | -          |
| 4.5       | Hybrid Downsampling     | SGD       | 64x64      | 100    | No               | -          |
| 5         | Block                   | SGD       | 64x64      | 100    | No               | -          |
| 6         | Residual Block          | SGD       | 64x64      | 100    | No               | -          |
| 7         | Residual Block V2       | SGD       | 64x64      | 100    | Yes              | -          |
| 8         | VGGNeXt                 | SGD       | 64x64      | 150    | Yes              | 10 epochs  |

