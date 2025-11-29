import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import hydra
import random
import json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from torch.quasirandom import SobolEngine
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose


from smallresnet import SmallResNet
from fashionmnist import FashionMNIST
from bayesian_optimizer import BayesianOptimizer
from utils import sobol_lrs

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: DictConfig):
    ds = FashionMNIST(batch_size=cfg.batch_size, val_ratio=cfg.val_ratio)

    train_loader = ds.get_train_loader()
    val_loader = ds.get_val_loader()

    return train_loader, val_loader

def accuracy(pred, true):
    class_index_pred = pred.argmax(dim=1)
    correct = (class_index_pred == true).sum().item()
    return correct / true.size(0)

def run_optimizing_function(device, lr, train_loader, val_loader, epochs):
    # initialize model
    model = SmallResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    val_losses = []
    train_losses = []
    avg_train_loss = 0.0
    avg_train_acc = 0.0
    avg_val_loss = 0.0
    avg_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loop = tqdm(train_loader, desc=f"Train (lr={lr:.1e})")
        total_train_loss = 0.0
        total_train_acc = 0.0
        for images, labels in train_loop:
            images = images.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_train_loss += loss.item()
            total_train_acc += accuracy(outputs, labels)
    
            if not np.isfinite(loss.item()):
                print(f"[WARN] Loss is {loss.item()} at lr={lr:.1e}. Returning huge loss.")
                fake_val_loss = 10.0
                return -fake_val_loss, model
                
            loss.backward()
            optimizer.step()
    
            train_acc = accuracy(outputs, labels)
            train_loop.set_postfix({"Loss": loss.item(),
                                    "Accuracy": train_acc})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
            
        print("Validating")
        model.eval()
        with torch.no_grad():
            # Validation Loss
            total_val_loss = 0.0
            total_val_acc = 0.0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                total_val_loss += criterion(outputs, labels).item()
                total_val_acc += accuracy(outputs, labels)
    
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_acc = total_val_acc / len(val_loader)
            if not np.isfinite(avg_val_loss):
                print(f"[WARN] avg_val_loss is {avg_val_loss} at lr={lr:.1e}. Forcing to 10.0")
                avg_val_loss = 10.0
            print("--> Validation-Loss :%.4f & Validation-Accuracy: %.4f" % (avg_val_loss, avg_val_acc))

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    epochs_axis = np.arange(1, epochs + 1)  # [1, 2, ..., epochs]

    plt.figure(figsize=(8, 5))
    
    plt.plot(epochs_axis, train_losses, marker='o', label="Train Loss")

    plt.plot(epochs_axis, val_losses, marker='o', label="Validation Loss")

    #print("train_losses: "+str(train_losses))
    #print("val losses: "+str(val_losses))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(f"results/{lr}_epoch_losses.png", dpi=200)

    return -avg_val_loss, model


def run_single_bo_experiment(cfg: DictConfig, seed: int):
    print(f"\n========== Seed {seed} ==========")
    set_global_seed(seed)

    device = torch.device(cfg.device)

    train_loader, val_loader = build_dataloaders(cfg)

    # --- GP & BO setup ---
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
    )

    bound = [cfg.bound.low, cfg.bound.high]

    bo = BayesianOptimizer(
        f=lambda lr: run_optimizing_function(device, lr, train_loader, val_loader, epochs=cfg.epochs),
        gp=gp,
        mode="log",
        bound=bound,
        path=cfg.result_path,
        size_search_space=250
    )

    n_total = cfg.n_total
    n_sobol = cfg.n_sobol
    n_iter = n_total - n_sobol

    sobol_init_lrs = sobol_lrs(n_sobol, bound, seed=seed)
    print("Sobol-Start-LRs:", sobol_init_lrs)

    x_max = 0

    for lr in sobol_init_lrs:
        x_max, best_return_x, best_return_param = bo.eval(1, lr)
        print(f"Learning rate {lr} from Sobol leads to better x_max: {x_max}")

    x_max, found_lr, best_model = bo.eval(n_iter=n_iter, init_x_max=x_max)
    
    print(f"[Seed {seed}] --> Found best learning rate at: {found_lr:.6f}")

    bo.plot_all_in_one(cols=1, seed=seed, save=True, show=False)

    bo.plot_lr_vs_loss(seed=seed, save=True, show=False)

    bo.plot_convergence(seed=seed, save=True, show=False)
    
    best_y = max([pt[1] for pt in bo.dataset])
    return {
        "seed": seed,
        "found_lr": float(found_lr),
        "best_y": float(best_y),
    }

def main():
    with initialize(version_base=None, config_path="conf"):
         cfg = compose(config_name="config")

    print("Config:\n", OmegaConf.to_yaml(cfg))
    
    results = []
    
    for seed in cfg.seeds:
        res = run_single_bo_experiment(cfg, seed)
        results.append(res)

    
    out_path = "results/bo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {out_path}")
    print("All seeds done.")



if __name__ == "__main__":
    main()
