#!/usr/bin/env python
import os
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import click
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tabpfn import TabPFNClassifier
import optuna
from utils import split_data  # Ensure split_data is defined in utils.py

# Setup logging to file and console.
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    
    # Set logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# Define the MLP model.
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_rate: float):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Save prediction results and classification metrics to CSV.
def save_results(model_name: str, y_true, y_pred, output_dir: str, dataset_name: str):
    results_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    })
    results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_predictions.csv")
    results_df.to_csv(results_path, index=False)
    
    metrics = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_metrics.csv")
    metrics_df.to_csv(metrics_path)

# Save the data splits so they can be re-used later.
def save_data_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir: str):
    np.savez(os.path.join(output_dir, "data_splits.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

# Train and evaluate a KNN model.
def train_knn(X_train, y_train, X_val, y_val, X_test, y_test, neighbors: int, gridsearch: bool, output_dir: str):
    logger = logging.getLogger()
    logger.info("Starting training for KNN...")
    
    if gridsearch:
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
        knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    else:
        knn = KNeighborsClassifier(n_neighbors=neighbors)
    
    knn.fit(X_train, y_train)
    
    for X, y, name in [(X_val, y_val, "validation"), (X_test, y_test, "test")]:
        y_pred = knn.predict(X)
        save_results("KNN", y, y_pred, output_dir, name)
        acc = accuracy_score(y, y_pred)
        logger.info(f"KNN {name.capitalize()} Accuracy: {acc:.4f}")

# The objective function for hyperparameter optimization using Optuna.
def objective(trial, X_train, y_train, X_val, y_val, device):
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    dr = trial.suggest_uniform("dr", 0.2, 0.5)
    epochs = 20  # Fixed for each trial; you may also make this a parameter.
    
    model = MLP(X_train.shape[1], hidden_size, len(np.unique(y_train)), dr).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=32
    )
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        best_val_acc = max(best_val_acc, val_acc)
    return best_val_acc

# Train and evaluate the MLP model.
def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
              hidden_size: int, lr: float, dr: float, epochs: int,
              gridsearch: bool, output_dir: str):
    logger = logging.getLogger()
    logger.info("Starting training for MLP...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameter optimization if requested.
    if gridsearch:
        logger.info("Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, device), n_trials=20)
        logger.info(f"Best hyperparameters: {study.best_params}")
        hidden_size = study.best_params["hidden_size"]
        lr = study.best_params["lr"]
        dr = study.best_params["dr"]
    
    # Prepare the model, loss, optimizer, and dataloaders.
    model = MLP(X_train.shape[1], hidden_size, len(np.unique(y_train)), dr).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=32
    )
    
    # Training loop.
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"MLP Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    # Evaluation on validation and test sets.
    model.eval()
    for X, y, name in [(X_val, y_val, "validation"), (X_test, y_test, "test")]:
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float).to(device)
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
        y_pred_np = y_pred.cpu().numpy()
        save_results("MLP", y, y_pred_np, output_dir, name)
        acc = accuracy_score(y, y_pred_np)
        logger.info(f"MLP {name.capitalize()} Accuracy: {acc:.4f}")

# Train and evaluate the TabPFN classifier.
def train_tabpfn(X_train, y_train, X_val, y_val, X_test, y_test, output_dir: str):
    logger = logging.getLogger()
    logger.info("Starting training for TabPFN...")
    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    for X, y, name in [(X_val, y_val, "validation"), (X_test, y_test, "test")]:
        y_pred = model.predict(X)
        save_results("TabPFN", y, y_pred, output_dir, name)
        acc = accuracy_score(y, y_pred)
        logger.info(f"TabPFN {name.capitalize()} Accuracy: {acc:.4f}")

# Main CLI entry point.
@click.command()
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input AnnData file.")
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(), help="Output directory.")
@click.option("--neighbors", default=3, help="Number of neighbors for KNN.", type=int)
@click.option("--hidden-size", default=64, help="Hidden layer size for MLP.", type=int)
@click.option("--lr", default=0.01, help="Learning rate for MLP.", type=float)
@click.option("--dr", default=0.5, help="Dropout rate for MLP.", type=float)
@click.option("--epochs", default=20, help="Number of training epochs for MLP.", type=int)
@click.option("--gridsearch", is_flag=True, help="Perform hyperparameter search for MLP.")


def main(input_path, output_dir, neighbors, hidden_size, lr, dr, epochs, gridsearch):
    setup_logging(output_dir)
    logger = logging.getLogger()
    logger.info("Starting training pipeline...")
    
    # Read the AnnData file and split the data.
    adata = ad.read_h5ad(input_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(adata)
    save_data_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir)
    
    # Train and evaluate each model.
    logger.info("Training and evaluating KNN...")
    train_knn(X_train, y_train, X_val, y_val, X_test, y_test, neighbors, gridsearch, output_dir)
    
    logger.info("Training and evaluating MLP...")
    train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, hidden_size, lr, dr, epochs, gridsearch, output_dir)
    
    logger.info("Training and evaluating TabPFN...")
    train_tabpfn(X_train, y_train, X_val, y_val, X_test, y_test, output_dir)
    
    logger.info("Training pipeline complete.")

if __name__ == "__main__":
    main()
