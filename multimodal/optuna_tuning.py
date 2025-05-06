import argparse
import os
import torch
import torch.nn.functional as F
import yaml
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import optuna
from model import MultiModalModel
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
    # Load base configuration from file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Override hyperparameters with Optuna suggestions
    config["training"]["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
    config["model"]["fused_dim"] = trial.suggest_int("fused_dim", 64, 256, step=64)
    config["model"]["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
    config["training"]["contrastive_loss_weight"] = trial.suggest_uniform("contrastive_loss_weight", 0.0, 1.0)
    config["model"]["mode"] = trial.suggest_categorical("mode", ["both"])
    config["model"]["num_heads"] = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    config["model"]["num_self_attention_layers"] = trial.suggest_int("num_self_attention_layers", 1, 3)
    dropout_prob = trial.suggest_uniform("dropout_prob", 0.1, 0.9)

    # For fast tuning, reduce the number of epochs.
    tuning_epochs = 30
    config["training"]["epochs"] = tuning_epochs

    # Load data using your utility functions.
    train_df, val_df, test_df, img_feat_cols = load_and_prepare_data(config)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        config["data"]["smiles_column"], img_feat_cols, config["data"]["label_column"],
        config["training"]["batch_size"]
    )

    # Instantiate the model with the current hyperparameters.
    model = MultiModalModel(
        img_dim=len(img_feat_cols),
        chem_model_name=config["model"]["chem_model_name"],
        fused_dim=config["model"]["fused_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        num_self_attention_layers=config["model"]["num_self_attention_layers"],
        num_classes=config["model"]["num_classes"],
        task=config["model"]["task"],
        freeze_chem_encoder=config["model"]["freeze_chem_encoder"],
        dropout_prob=dropout_prob
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tuning_epochs)

    best_val_loss = float('inf')
    mode = config["model"].get("mode", "both")

    # --- Training Loop ---
    for epoch in range(tuning_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)
            if config["model"]["task"] == 'classification':
                loss_pred = F.cross_entropy(preds, batch["label"].to(device))
            else:
                loss_pred = F.mse_loss(preds.squeeze(), batch["label"].float().to(device))

            if mode == "both":
                loss_cont = symmetric_contrastive_loss(img_emb, chem_emb)
            else:
                loss_cont = 0.0

            loss = loss_pred + config["training"]["contrastive_loss_weight"] * loss_cont
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)
                if config["model"]["task"] == 'classification':
                    loss_pred = F.cross_entropy(preds, batch["label"].to(device))
                    preds_labels = preds.argmax(dim=1).cpu().numpy()
                else:
                    loss_pred = F.mse_loss(preds.squeeze(), batch["label"].float().to(device))
                    preds_labels = preds.squeeze().cpu().numpy()
                if mode == "both":
                    loss_cont = symmetric_contrastive_loss(img_emb, chem_emb)
                else:
                    loss_cont = 0.0
                loss = loss_pred + config["training"]["contrastive_loss_weight"] * loss_cont
                val_loss += loss.item()

                all_preds.extend(preds_labels)
                all_targets.extend(batch["label"].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)

        # Compute additional evaluation metrics on the validation set.
        val_accuracy = accuracy_score(all_targets, all_preds)
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
        per_class_f1 = f1_score(all_targets, all_preds, average=None).tolist()

        # Log metrics as trial attributes.
        trial.set_user_attr("epoch_{}".format(epoch+1), {
            "val_loss": avg_val_loss,
            "accuracy": val_accuracy,
            "weighted_f1": weighted_f1,
            "per_class_f1": per_class_f1
        })

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Also log overall metrics from the final epoch.
    trial.set_user_attr("final_accuracy", val_accuracy)
    trial.set_user_attr("final_weighted_f1", weighted_f1)
    trial.set_user_attr("final_per_class_f1", per_class_f1)

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

    # Optionally, save the best hyperparameters to a YAML file
 # Load the original config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Apply best hyperparameters from Optuna to the config
    best_hyperparams = best_trial.params

    config["training"]["learning_rate"] = best_hyperparams["learning_rate"]
    config["model"]["fused_dim"] = best_hyperparams["fused_dim"]
    config["model"]["hidden_dim"] = best_hyperparams["hidden_dim"]
    config["training"]["contrastive_loss_weight"] = best_hyperparams["contrastive_loss_weight"]
    config["model"]["mode"] = best_hyperparams["mode"]
    config["model"]["num_heads"] = best_hyperparams["num_heads"]
    config["model"]["num_self_attention_layers"] = best_hyperparams["num_self_attention_layers"]
    config["model"]["dropout_prob"] = best_hyperparams["dropout_prob"]

    # Save the complete updated config with Optuna's best hyperparameters
    with open("best_config.yaml", "w") as f:
        yaml.dump(config, f)

    print("Updated config saved to best_config.yaml")
