# AUTOML-APPLICATION

This project implements Bayesian Optimization (BO) to automatically optimize the learning rate of a convolutional neural network trained on Fashion-MNIST.

The optimization uses:

- Sobol sequence for the initial design (quasi-random exploration),

- a Gaussian Process (GP) as the surrogate model,

- Weighted Expected Improvement (WEI) as the acquisition function,

- PyTorch for training a custom Small ResNet,

The goal is to find the learning rate that minimizes the validation loss after a fixed number of epochs.

All BO iterations are visualized and stored for later analysis.

## Installation

Create and activate a virtual environment. Then install:

```
pip install -r requirements.txt
```

## Running the experiment

Please set up the parameters in the **conf/config.yaml**. 

Important: Please choose the right device!

After that run:

```
python3 main.py
```

## Project structure

```
project_root/
│
├── bayesian_optimizer.py      # GP-based BO implementation incl. WEI
├── fashionmnist.py            # For loading the dataset
├── smallresnet.py             # The computer vision model based on a ResNet
├── utils.py                   # Sobol sequence function is there.
├── main.py                    # Entrypoint
│
├── conf/
│   └── config.yaml            # experiment settings (bounds, epochs, seeds, device, ...)
│
├── results/
│   └── experiment/                # plots + logs for each seed with an epoch of 10 for 5 different seeds
│        └── bo_results.json     # includes the results per seed. (best learning rate, best loss)
│        └── **_all_iterations.png     # includes the graphs for each bayesian optimization. 
│        └── **_epoch_losses.png     # Loss per epoch
│        └── **_iterationvsbestsofar.png     # best loss per iteration of the bo.
│
└── README.md
```

## Methods 

### ResNet

The given small ResNet is based on an article on Medium by Sai Nihith (Link is in smallresnet.py). I just added some Batch Normalizations and Pooling to some pooling layers to fasten the training process.

### Goal

The goal of BO is:

f(lr)=−Validation Loss(lr)

The negative sign converts the minimization problem into a maximization problem, which was easier to use the WEI formula for.

### SOBOL Sequence

The SOBOL Sequence is generated within the learning-rate bounds. The bounds are saved in **conf/config.yaml**.

## Results

I added some thoughts and notes to this project. 

### Bayesian Optimization

After implementing the bayesian optimization, I set the following parameters for visualizations.

```
config.yaml
# training / bo parameter
epochs: 10
seeds: [0, 1, 2, 3, 4]  
n_total: 2        
n_sobol: 1    
bound:
  low: 1e-4
  high: 1e-1



batch_size: 64
val_ratio: 0.2  # ratio between training and validation
```

The following visualizations were made:

- GP posterior + uncertainty + WEI per BO iteration (Seed_x_all_iterations.png)
- Epoch loss curve (train/val) for each LR evaluation (Seed_X_{LR}_epoch_losses.png)
- The best learning rate per BO iteration (Seed_x_iterationsvsbestsofar.png)


## Interpretation

Although Bayesian Optimization successfully adapts its acquisition function and explores promising regions of the learning-rate space, the obtained results show notable variability across seeds. The main observations are:

- Sobol initialization strongly influences the outcome:
  Each seed produces a different Sobol sequence, leading to different initial evaluations and therefore different optimization trajectories.
- 3 of 5 results of different seeds have the best learning rate of 0.1. That is the upper bound.
  This indicates that the optimal region may lie beyond the current bound.
- The remaining two seeds exhibit noisy behavior and high posterior uncertainty.
  It suggests that the number of evaluations may be too small to stabilize the surrogate function.

To improve the stability and quality of the optimization results, I would recommend the following:

- Increase the number of training epochs (reducing the noise in validation loss)
- Evaluate more seeds
- Experiment with different GP kernels
  So far, only a single kernel (Constant × RBF) has been applied.
  Alternatives may capture the shape of the loss landscape better.
- Expand the learning-rate bound
- Adjust the WEI α parameter.
  With balancing the exploration and exploitation it may stabilize the optimization.


