import argparse
import os
import torch
import torch.nn.functional as F
import yaml
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import optuna
import wandb  # Added Weights & Biases
from model_v2 import MultiModalModel
from utils import load_and_prepare_data, get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def symmetric_contrastive_loss(img_emb, chem_emb, temperature=0.07):
    logits_img2chem = F.normalize(img_emb, dim=-1) @ F.normalize(chem_emb, dim=-1).T / temperature
    logits_chem2img = logits_img2chem.T
    labels = torch.arange(len(logits_img2chem)).to(logits_img2chem.device)
    loss_img2chem = F.cross_entropy(logits_img2chem, labels)
    loss_chem2img = F.cross_entropy(logits_chem2img, labels)
    return (loss_img2chem + loss_chem2img) / 2

def objective(trial):
    # Load base configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Tune Hyperparameters ---
    config["training"]["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
    config["model"]["fused_dim"] = trial.suggest_int("fused_dim", 64, 256, step=64)
    config["model"]["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
    config["training"]["contrastive_loss_weight"] = trial.suggest_uniform("contrastive_loss_weight", 0.0, 1.0)
    config["model"]["mode"] = "both"

    # Fusion Architecture
    config["model"]["fusion_type"] = "transformer"
    config["model"]["fusion_agg"] = trial.suggest_categorical("fusion_agg", ["cls", "mean", "max"])
    config["model"]["num_heads"] = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    config["model"]["num_self_attention_layers"] = trial.suggest_int("num_self_attention_layers", 1, 3)
    dropout_prob = trial.suggest_uniform("dropout_prob", 0.1, 0.9)

    # Reduce epochs for tuning
    tuning_epochs = 30
    config["training"]["epochs"] = tuning_epochs

    # --- Initialize Weights & Biases ---
    wandb.init(project="multimodal-hyperparameter-tuning", name=f"trial_{trial.number}", config=config)

    # --- Load Data ---
    train_df, val_df, test_df, img_feat_cols = load_and_prepare_data(config)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        config["data"]["smiles_column"], img_feat_cols, config["data"]["label_column"],
        config["training"]["batch_size"]
    )

    # --- Model Initialization ---
    model = MultiModalModel(
        img_dim=len(img_feat_cols),
        chem_model_name=config["model"]["chem_model_name"],
        chem_model_type=config["model"]["chem_model_type"],
        fused_dim=config["model"]["fused_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        num_self_attention_layers=config["model"]["num_self_attention_layers"],
        num_classes=config["model"]["num_classes"],
        task=config["model"]["task"],
        freeze_chem_encoder=config["model"]["freeze_chem_encoder"],
        dropout_prob=dropout_prob,
        fusion_type=config["model"]["fusion_type"],
        fusion_agg=config["model"]["fusion_agg"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tuning_epochs)

    best_val_loss = float('inf')
    mode = config["model"]["mode"]

    # --- Training Loop ---
    for epoch in range(tuning_epochs):
        model.train()
        total_loss, total_pred_loss, total_cont_loss = 0.0, 0.0, 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)

            loss_pred = F.cross_entropy(preds, batch["label"].to(device)) if config["model"]["task"] == 'classification' \
                        else F.mse_loss(preds.squeeze(), batch["label"].float().to(device))

            loss_cont = symmetric_contrastive_loss(img_emb, chem_emb) if mode == "both" else 0.0
            loss = loss_pred + config["training"]["contrastive_loss_weight"] * loss_cont

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += loss_pred.item()
            total_cont_loss += loss_cont.item() if isinstance(loss_cont, torch.Tensor) else 0.0

        scheduler.step()

        # --- Log Training Metrics to wandb ---
        wandb.log({
            "train_total_loss": total_loss / len(train_loader),
            "train_pred_loss": total_pred_loss / len(train_loader),
            "train_contrastive_loss": total_cont_loss / len(train_loader),
            "epoch": epoch + 1
        })

        # --- Validation Loop ---
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []

        with torch.no_grad():
            for batch in val_loader:
                preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)
                loss_pred = F.cross_entropy(preds, batch["label"].to(device)) if config["model"]["task"] == 'classification' \
                            else F.mse_loss(preds.squeeze(), batch["label"].float().to(device))

                loss_cont = symmetric_contrastive_loss(img_emb, chem_emb) if mode == "both" else 0.0
                loss = loss_pred + config["training"]["contrastive_loss_weight"] * loss_cont
                val_loss += loss.item()

                preds_labels = preds.argmax(dim=1).cpu().numpy() if config["model"]["task"] == 'classification' \
                              else preds.squeeze().cpu().numpy()
                val_preds.extend(preds_labels)
                val_targets.extend(batch["label"].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)

        val_accuracy = accuracy_score(val_targets, val_preds)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted')

        # --- Log Validation Metrics to wandb ---
        wandb.log({
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": weighted_f1,
        })

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    wandb.finish()

    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for multimodal model")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials to run")
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("Best trial user attributes:", best_trial.user_attrs)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- Save Updated Config ---
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    best_hyperparams = best_trial.params

    # Apply tuned hyperparameters to full config
    config["training"]["learning_rate"] = best_hyperparams["learning_rate"]
    config["model"]["fused_dim"] = best_hyperparams["fused_dim"]
    config["model"]["hidden_dim"] = best_hyperparams["hidden_dim"]
    config["training"]["contrastive_loss_weight"] = best_hyperparams["contrastive_loss_weight"]
    config["model"]["fusion_agg"] = best_hyperparams["fusion_agg"]
    config["model"]["num_heads"] = best_hyperparams["num_heads"]
    config["model"]["num_self_attention_layers"] = best_hyperparams["num_self_attention_layers"]
    config["model"]["dropout_prob"] = best_hyperparams["dropout_prob"]

    # Save the updated full config file
    with open("best_config.yaml", "w") as f:
        yaml.dump(config, f)

    print("Updated config saved to best_config.yaml")
