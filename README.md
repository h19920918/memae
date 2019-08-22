# Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection (MemAE)

** !!! This repo. is not official and not perpectly implemented. !!! **

You can see the paper in https://arxiv.org/pdf/1904.02639.pdf.

It is partly implemented since it is a part of my current project.

# Requirements
* Python 3.6.4
* Pytorch >= 1.0.0
* tensorboardX >= 1.6
* tqdm >= 1.6

# How to use

## Train

To train the model,
```shell
python main.py \
  --train
  --num-istances [INT] \
  --num-classes [INT] \
  --num-memories [INT] \
  --addressing ['soft', 'sparse'] \
  --log-dir=$DIR
```

You can see other configuration in 'main.py'.

## Visualize
```shell
python main.py --visualize --ckpt [TRAINED_MODEL]
```

## Test
```shell
python main.py --test --ckpt [TRAINED_MODEL]
