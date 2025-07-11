# Code for Visual Perturbation for Text-based Person Search

## Prerequest

1. Main dependecies

    ```shell
        torch==2.5.0
        torchvision==0.20.0
        cuda==12.1
        detectron2==0.6
    ```

2. Preparing data

    * Download [CUHK-SYSU-TBPS](https://github.com/Dacun/Text-based-Person-Search), [PRW-TBPS](https://github.com/Dacun/Text-based-Person-Search) for text-based person search.

    * Create folder ```Data/``` and place the data as

        ```shell
                Data
                ├── cuhk_sysu
                │   ├── annotation
                │   ├── Image
                │   ├── CUHK-SYSU-TBPS_test
                │   ├── CUHK-SYSU-TBPS_train
                │   └── README.txt
                └── PRW
                    ├── annotations
                    ├── frames
                    ├── PRW-TBPS_test
                    ├── PRW-TBPS_train
                    └── ......
        ```


## Training

* Train any model with the config

    ```shell
    python tools/train_ps_net.py --config-file ${file_path} --resume
    ```

* We recommend to use [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor the training process:

    ```shell
    tensorboard --logdir outputs/${model_output}$
    ```

## Inference

* Test any model with the config

    ```shell
    python tools/train_ps_net.py --config-file ${file_path} --resume --eval-only
    ```

## Code structure

* The overall structure of the code is based on [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html). This repository is organized as

    ```shell
    ViPer
    ├── configs # Where model configuration files saved
    ├── Data # Where data and model weights saved 
    ├── LICENSE
    ├── outputs # Where training logs and checkpoints saved
    ├── psd2 # The main code
    │   ├── checkpoint # Automatic checkpoint tool
    │   ├── config # Definition of the basic configuration for all models
    │   ├── data # Reading and augmenting data
    │   ├── engine # The main training loop
    │   ├── evaluation # Evaluators for testing the model 
    │   ├── __init__.py
    │   ├── layers # Widely-used neural network modules
    │   ├── modeling # Definition of models
    │   │   ├── backbone # Vision backbone models
    │   │   ├── meta_arch
    │   │   │   ├── person_search # Person search models
    │   │   │   └── ......
    │   │   ├── proposal_generator # RPN and its variants
    │   │   ├── roi_heads # Prediction modules
    │   │   └── ......
    │   ├── model_zoo
    │   ├── solver # Optimizers and lr schedulers
    │   ├── structures # Commonly used data structures
    │   └── utils # Commonly used tools, e.g. logging, visualization and distributed communication.
    ├── README.md # This file
    └── tools # Train/test interface
        └── train_ps_net.py

    ```

## Acknowledgement

This code is greatly inspired by [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).
