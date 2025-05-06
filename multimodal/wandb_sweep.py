import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
import wandb
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from model_v2 import MultiModalModel
from utils import load_and_prepare_data, get_dataloaders
from sklearn.metrics import classification_report
from collections import Counter
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################
# 1) Define SupConLoss (Your Code)
##################################
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: 
    https://arxiv.org/pdf/2004.11362.pdf.
    Also degenerates to SimCLR if no labels provided.
    """
    def __init__(self, temperature: float, contrast_mode: str='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels=None):
        """
        out0, out1: shape [batch_size, embed_dim]
        labels (optional): shape [batch_size], same class => positives.
        Returns: (loss, clean_logits, label_indices) for debugging
        """
        device = out0.device
        
        # 1) L2 Normalize
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)
        
        # 2) Stack => shape [batch_size, 2, embed_dim]
        features = torch.stack([out0, out1], dim=1)
        mask = None

        if len(features.shape) < 3:
            raise ValueError('features must be [bsz, n_views, ...]')

        # batch_size
        bsz = features.shape[0]

        # If no labels => default to identity mask (SimCLR)
        if labels is None:
            mask = torch.eye(bsz, dtype=torch.float32).to(device)
        else:
            # Turn labels into shape [bsz, 1], 
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        # We flatten the n_views dimension
        contrast_count = features.shape[1]  # e.g. 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # "all" => each view is an anchor; "one" => only first view is anchor
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown contrast_mode={self.contrast_mode}')
        
        # Compute logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask out self-contrast
        logits_mask = torch.ones_like(mask)
        idx = torch.arange(bsz * anchor_count).to(device).view(-1, 1)
        logits_mask = torch.scatter(logits_mask, 1, idx, 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # average over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, bsz).mean()

        # Additional "clean_logits" for debugging
        # Remove self-sim
        batch_size_2 = bsz * anchor_count
        clean_logits = exp_logits[~torch.eye(batch_size_2).bool()].view(batch_size_2, -1)
        labels_idx = torch.arange(bsz, device=device)
        
        return loss, clean_logits, labels_idx

def symmetric_contrastive_loss(img_emb, chem_emb, temperature=0.07):
    logits_img2chem = F.normalize(img_emb, dim=-1) @ F.normalize(chem_emb, dim=-1).T / temperature
    logits_chem2img = logits_img2chem.T
    labels = torch.arange(len(logits_img2chem)).to(logits_img2chem.device)
    loss_img2chem = F.cross_entropy(logits_img2chem, labels)
    loss_chem2img = F.cross_entropy(logits_chem2img, labels)
    return (loss_img2chem + loss_chem2img) / 2

def train():
    # 1) Load base config for data, default hyperparameters, etc.
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # 2) Initialize wandb (this will read hyperparams from the sweep)
    wandb.init(project="multimodal-hyperparameter-tuning", entity="benjamin-frey-uppsala-universitet", config=base_config)

    # 3) Merge wandb.config into base_config
    final_config = base_config
    for key in wandb.config.keys():
        if key in ["num_views", "learning_rate", "contrastive_loss_weight", "supcon_temp", "contrastive_loss_type",
                   "noise_std", "col_drop_prob", "smiles_aug_prob"]:
            final_config["training"][key] = wandb.config[key]
        elif key in ["fused_dim", "hidden_dim", "num_heads", "num_self_attention_layers", 
                     "dropout_prob", "fusion_agg", "fusion_type"]:
            final_config["model"][key] = wandb.config[key]

    config = final_config

    # 4) Load data
    num_views = config["training"].get("num_views", 0)
    train_df, val_df, test_df, img_feat_cols = load_and_prepare_data(config)
    
    # Instead of passing config directly, 
    # we extract relevant augmentation args and pass them to get_dataloaders:
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        smiles_col=config["data"]["smiles_column"],
        img_feat_cols=img_feat_cols,
        label_col=config["data"]["label_column"],
        batch_size=config["training"]["batch_size"],
        noise_std=config["training"].get("noise_std", 0.01),
        col_drop_prob=config["training"].get("col_drop_prob", 0.05),
        smiles_aug_prob=config["training"].get("smiles_aug_prob", 0.5),
        augment= True if num_views > 0 else False,
        num_views = num_views
    )

    # 5) Initialize the model with updated hyperparams
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
        fusion_type = config["model"]["fusion_type"],
        freeze_chem_encoder=config["model"]["freeze_chem_encoder"],
        dropout_prob=config["model"]["dropout_prob"],
        #fusion_agg=config["model"]["fusion_agg"]
    ).to(device)

    label_counts = Counter(train_df[config["data"]["label_column"]].values)
    total = sum(label_counts.values())
    num_classes = config["model"]["num_classes"]

    # Compute inverse frequency weights (can adjust as needed)
    weights = [total / label_counts[i] if i in label_counts else 1.0 for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])

    supcon = SupConLoss(temperature=config["training"].get("supcon_temp", 0.07))

    best_val_loss = float('inf')
    mode = config["model"].get("mode", "both")

    # 6) Training Loop
    start_time = time.time()  # <-- ADD THIS

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss, total_pred_loss, total_contr_loss = 0.0, 0.0, 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)

            # 7A) Normal supervised (classification/regression) loss
            if config["model"]["task"] == "classification":
                loss_pred = F.cross_entropy(input = preds, target = batch["label"].to(device), weight = class_weights)
            else:
                loss_pred = F.mse_loss(preds.squeeze(), batch["label"].float().to(device))

            # 7B) If mode == "both", apply chosen contrastive
            if mode == "both":
                c_loss_type = config["training"].get("contrastive_loss_type", "none")
                if c_loss_type == "sym":
                    cont_loss = symmetric_contrastive_loss(img_emb, chem_emb, temperature=0.07)
                elif c_loss_type == "supcon":
                    # If classification => pass labels, else pass None
                    supcon_labels = batch["label"].to(device) if config["model"]["task"] == "classification" else None
                    cont_loss, _, _ = supcon(img_emb, chem_emb, labels=supcon_labels)
                else:
                    cont_loss = 0.0
            else:
                # If not "both" => no contrastive
                cont_loss = 0.0

            # Weighted sum
            c_weight = config["training"].get("contrastive_loss_weight", 0.0)
            loss = loss_pred + c_weight * cont_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += loss_pred.item()
            if isinstance(cont_loss, torch.Tensor):
                total_contr_loss += cont_loss.item()

        scheduler.step()

        # Log training stats
        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_pred_loss": total_pred_loss / len(train_loader),
            "train_contr_loss": total_contr_loss / len(train_loader),
            "epoch": epoch + 1
        })

        # 7) Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        # For AUPRC
        val_probs_list = []  # store predicted probabilities for AUPRC
        with torch.no_grad():
            for batch in val_loader:
                preds, img_emb, chem_emb = model(batch["img_feat"].to(device), batch["smiles"], mode=mode)

                # Supervised loss
                if config["model"]["task"] == "classification":
                    loss_pred = F.cross_entropy(preds, batch["label"].to(device))
                    pred_labels = preds.argmax(dim=1).cpu().numpy()

                    # Also get probabilities for AUPRC
                    probs = F.softmax(preds, dim=1).cpu().numpy()  # (batch, num_classes)
                    val_probs_list.append(probs)
                else:
                    loss_pred = F.mse_loss(preds.squeeze(), batch["label"].float().to(device))
                    pred_labels = preds.squeeze().cpu().numpy()

                if (mode == "both") and (c_loss_type != "none"):
                    if c_loss_type == "sym":
                        cont_val = symmetric_contrastive_loss(img_emb, chem_emb, temperature=0.07)
                    elif c_loss_type == "supcon":
                        supcon_labels = batch["label"].to(device) if (config["model"]["task"] == "classification") else None
                        sc_loss, _, _ = supcon(img_emb, chem_emb, labels=supcon_labels)
                        cont_val = sc_loss
                    else:
                        cont_val = 0.0
                else:
                    cont_val = 0.0

                c_weight = config["training"].get("contrastive_loss_weight", 0.0)
                batch_loss = loss_pred + c_weight * cont_val
                val_loss += batch_loss.item()

                val_preds.extend(pred_labels)
                val_targets.extend(batch["label"].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Evaluate classification metrics
        if config["model"]["task"] == "classification":
            val_accuracy = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='weighted')

            # AUPRC
            num_classes = config["model"]["num_classes"]
            val_probs_cat = None
            if num_classes == 2:
                # binary => we take prob of class 1
                val_probs_cat = torch.tensor([], dtype=torch.float)
                # combine
                val_probs_cat = torch.tensor([], dtype=torch.float)
                for arr in val_probs_list:
                    val_probs_cat = torch.cat((val_probs_cat, torch.tensor(arr[:, 1])), dim=0)
                val_probs_cat = val_probs_cat.numpy()
                
                from sklearn.metrics import average_precision_score
                val_auprc = average_precision_score(val_targets, val_probs_cat)
            else:
                # multi-class => macro average
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import average_precision_score

                # Concatenate predictions
                all_probs = []
                for arr in val_probs_list:
                    all_probs.append(arr)
                val_probs_cat = torch.tensor([], dtype=torch.float)
                val_probs_cat = torch.tensor([])  # better do this in python
                import numpy as np
                val_probs_cat = np.concatenate(val_probs_list, axis=0)  # shape [N, num_classes]

                # binarize targets
                if isinstance(val_targets, list):
                    val_targets_np = np.array(val_targets)
                else:
                    val_targets_np = val_targets
                # classes are [0..num_classes-1]
                y_bin = label_binarize(val_targets_np, classes=range(num_classes))
                val_auprc = average_precision_score(y_bin, val_probs_cat, average="macro")
            
            wandb.log({
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "val_auprc": val_auprc,
                "epoch": epoch + 1
            })


            val_preds_np = np.array(val_preds)
            val_targets_np = np.array(val_targets)

                # Get per-class metrics
            class_report = classification_report(
                    val_targets_np,
                    val_preds_np,
                    output_dict=True,
                    zero_division=0  # avoid crashing if a class is missing
                )

                # Log individual class F1 scores and accuracies
            for class_id, metrics in class_report.items():
                if isinstance(metrics, dict):  # filter out 'accuracy', 'macro avg', etc.
                    wandb.log({
                        f"class_{class_id}_f1": metrics.get("f1-score", 0),
                        f"class_{class_id}_precision": metrics.get("precision", 0),
                        f"class_{class_id}_recall": metrics.get("recall", 0),
                        "epoch": epoch + 1
                    })
        else:
            # regression => no accuracy/f1
            wandb.log({
                "val_loss": avg_val_loss,
                "epoch": epoch + 1
            })

        # track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
    
    total_training_time = time.time() - start_time
    wandb.log({"total_training_time_sec": total_training_time})
    wandb.finish()

if __name__ == "__main__":
    train()
