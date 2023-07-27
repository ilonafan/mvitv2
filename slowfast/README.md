# Action Recognition with Noisy Labels

This is the code for the CV:HCI practical course with topic: Action Recognition with Noisy Labels

## Dependencies

We implement our methods by PyTorch on Quadro RTX 6000 and 8000 GPU. The environment is as bellow:

- [Python](https://python.org/), version = 3.10
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version = 1.12.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.6
- [Anaconda3](https://www.anaconda.com/)


## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md) and follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets in Kinetics format.

## Experiments

We verify the effectiveness of the PNP method and Robust Early-Learning method on simulated noisy datasets. 

In this repository, we provide the used labels in different protions. You should also download the NTU60 dataset and put them into the same folder of labels as the instructions in [DATASET.md](slowfast/datasets/DATASET.md).


Here is a training example: 
```bash
python tools/run_net.py \
  --cfg configs/Kinetics/MViTv2_S_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
```

To perform test, you can set the TRAIN.ENABLE to False, and do not forget to pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH.


## Visualization

To visualize the resultant embeddings of your model, you can first perform test and set the Task as t-sne and save the output csv file.

Here is an example: 
```bash
python tools/run_net.py \
  --cfg configs/Kinetics/MViTv2_S_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
  Task T-SNE\
```

Then you can run the script in "visualization" folder to visualiza them in 2d or 3D via t-SNE.

