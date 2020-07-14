# Orthant Based Proximal Stochastic Gradient Method for ℓ1-Regularized Optimization

PyTorch Implementation of non-convex experiments in "Orthant Based Proximal Stochastic Gradient Method for ℓ1-Regularized Optimization". 

## Abstract

Sparsity-inducing regularization problems are ubiquitous in machine learning applications, ranging from feature selection to model compression. In this paper, we present a novel stochastic method -- Orthant Based Proximal Stochastic Gradient Method (OBProx-SG) -- to solve perhaps the most popular instance, i.e., the l1-regularized problem. The OBProx-SG method contains two steps: (i) a proximal stochastic gradient step to predict a support cover of the solution; and (ii) an orthant step to aggressively enhance the sparsity level via orthant face projection. Compared to the state-of-the-art methods, e.g., Prox-SG, RDA and Prox-SVRG, the OBProx-SG not only converges to the global optimal solutions (in convex scenario) or the stationary points (in non-convex scenario), but also promotes the sparsity of the solutions substantially. Particularly, on a large number of convex problems, OBProx-SG outperforms the existing methods comprehensively in the aspect of sparsity exploration and objective values. Moreover, the experiments on non-convex deep neural networks, e.g., MobileNetV1 and ResNet18, further demonstrate its superiority by achieving the solutions of much higher sparsity without sacrificing generalization accuracy.

## Requirements

+ torch: 1.4.0 or higher
+ torchvision: 0.5.0 or higher

Please follow the instructions [here](<https://pytorch.org/get-started/locally/>) to install PyTorch.


## Set up Dataset

In `datasets.py`, `DATA_DIR`represents the path to dataset. Please replace this constant variable with the path to the dataset.

## Running Code

### Run all experiments

The scripts of running all non-convex experiments are provided in the `command.sh`. You can simply run the following command to test non-convex experiments:

```bash
bash command.sh
```

### Run specific experiment

+ optimizer: [ obproxsg | proxsg | rda | proxsvrg ]
+ model: [ mobilenetv1 | resnet18 ]
+ dataset_name: [ cifar10 | fashion_mnist ]


```bash
python run.py --optimizer <optimizer> \
              --model <model> \
              --dataset_name <dataset_name> \
              --lambda_ 0.0001 \
              --max_epoch 200 \
              --lr 0.1 \
              --batch_size 128
```

An example is:

```bash
python run.py --optimizer obproxsg \
              --model mobilenetv1 \
              --dataset_name cifar10 \
              --lambda_ 0.0001 \
              --max_epoch 200 \
              --lr 0.1 \
              --batch_size 128
```

## Evaluation

To evaluate our model, users can run the following command to evaluate the trained model. Make sure that the arguments match the training information of trained model.

```bash
python evaluate.py --model <model> \
                   --dataset_name <dataset_name> \
                   --lambda_ 0.0001 \
                   --ckpt <ckpt>
```

An example is:

```bash
python evaluate.py --model resnet18 \
                   --dataset_name cifar10 \
                   --lambda_ 0.0001 \
                   --ckpt checkpoints/obproxsg_plus_resnet18_cifar10_1.000000E-04.pt
```

## Acknowledgements
If you find the code or results useful, please kindly cite the following paper:
```
@article{chen2020orthant,
  title={Orthant Based Proximal Stochastic Gradient Method for $\ell_1 $-Regularized Optimization},
  author={Chen, Tianyi and Ding, Tianyu and Ji, Bo and Wang, Guanyi and Shi, Yixin and Yi, Sheng and Tu, Xiao and Zhu, Zhihui},
  journal={arXiv preprint arXiv:2004.03639},
  year={2020}
}
```
