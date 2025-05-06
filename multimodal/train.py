import argparse
import os
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from model_v2 import MultiModalModel
from utils import load_and_prepare_data, get_dataloaders
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def symmetric_contrastive_loss(img_emb, chem_emb, temperature=0.07):
    logits_img2chem = F.normalize(img_emb, dim=-1) @ F.normalize(chem_emb, dim=-1).T / temperature
    logits_chem2img = logits_img2chem.T
    labels = torch.arange(len(logits_img2chem)).to(logits_img2chem.device)

    loss_img2chem = F.cross_entropy(logits_img2chem, labels)
    loss_chem2img = F.cross_entropy(logits_chem2img, labels)

    return (loss_img2chem + loss_chem2img) / 2

def evaluate(model, loader, task, device, mode="both"):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            preds, _, _ = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)
            preds = (preds.argmax(dim=1).cpu().numpy() if task == "classification"
                     else preds.squeeze().cpu().numpy())
            all_preds.extend(preds)
            all_targets.extend(batch["label"].cpu().numpy())
    return all_preds, all_targets

def plot_loss_curves(train_losses, train_contrastive_losses, val_losses, save_path):
    """Plots and saves the training & validation loss curves."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker='o', linestyle='-')
    plt.plot(train_contrastive_losses, label="Train Contrastive Loss", marker='s', linestyle='--')
    plt.plot(val_losses, label="Validation Loss", marker='x', linestyle='-.')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def main(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_df, val_df, test_df, img_feat_cols = load_and_prepare_data(config)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        config["data"]["smiles_column"], img_feat_cols, config["data"]["label_column"],
        config["training"]["batch_size"]
    )

    model = MultiModalModel(
        img_dim=len(img_feat_cols),
        chem_model_name=config["model"]["chem_model_name"],
        chem_model_type = config["model"]["chem_model_type"],
        fused_dim=config["model"]["fused_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        num_self_attention_layers=config["model"]["num_self_attention_layers"],
        num_classes=config["model"]["num_classes"],
        task=config["model"]["task"],
        freeze_chem_encoder=config["model"]["freeze_chem_encoder"],
        dropout_prob=config["model"]["dropout_prob"]
    ).to(device)

    print("Model Architecture:\n", model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Trainable parameters:", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config["training"]["early_stopping_patience"]
    mode = config["model"].get("mode", "both")  

    # Track loss values
    train_losses, train_contrastive_losses, val_losses = [], [], []

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss, total_contr_loss = 0.0, 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)

            loss_pred = F.cross_entropy(preds, batch["label"].to(device)) \
                if config["model"]["task"] == 'classification' else \
                F.mse_loss(preds.squeeze(), batch["label"].float().to(device))

            if mode == "both":
                loss_cont = symmetric_contrastive_loss(img_emb, chem_emb)
            else:
                loss_cont = 0.0

            contr_loss = config["training"]["contrastive_loss_weight"] * loss_cont
            loss = loss_pred + contr_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contr_loss += contr_loss.item() if isinstance(contr_loss, torch.Tensor) else 0.0

        scheduler.step()

        train_losses.append(total_loss / len(train_loader))
        train_contrastive_losses.append(total_contr_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{config['training']['epochs']} - "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Train Contrastive Loss: {train_contrastive_losses[-1]:.4f}")

        # Validation loss calculation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"])
                loss_pred = F.cross_entropy(preds, batch["label"].to(device)) \
                    if config["model"]["task"] == 'classification' else \
                    F.mse_loss(preds.squeeze(), batch["label"].float().to(device))

                if mode == "both":
                    loss_cont = symmetric_contrastive_loss(img_emb, chem_emb)
                else:
                    loss_cont = 0.0

                loss = loss_pred + config["training"]["contrastive_loss_weight"] * loss_cont
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if config["training"]["early_stopping"] and patience_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    # Final Test Evaluation
    test_preds, test_labels = evaluate(model, test_loader, config["model"]["task"], device, mode=mode)

    metrics = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "f1_score": f1_score(test_labels, test_preds, average='weighted'),
        "mse": mean_squared_error(test_labels, test_preds) if config["model"]["task"] != 'classification' else None
    }

    results_dir = f"data/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # Save loss curves
    loss_plot_path = f"{results_dir}/loss_curve.png"
    plot_loss_curves(train_losses, train_contrastive_losses, val_losses, loss_plot_path)

    pd.DataFrame({"predictions": test_preds, "labels": test_labels}).to_csv(f"{results_dir}/predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(f"{results_dir}/metrics.csv", index=False)

    print(f"Results and metrics saved to {results_dir}, loss curve saved at {loss_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal training script with contrastive loss and cross-attention.")
    parser.add_argument("--config", default="best_config.yaml")
    args = parser.parse_args()
    main(args.config)
