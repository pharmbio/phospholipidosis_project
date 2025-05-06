import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SMILESEncoder(nn.Module):
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", freeze=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, smiles):
        inputs = self.tokenizer(
            smiles, return_tensors='pt', padding=True, truncation=True, max_length=128
        ).to(next(self.model.parameters()).device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return embeddings

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.cross_attn_img_to_chem = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn_chem_to_img = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)  # You can also expose this as a parameter if needed

    def forward(self, img_emb, chem_emb):
        img_emb = img_emb.unsqueeze(1)
        chem_emb = chem_emb.unsqueeze(1)

        img_attn, _ = self.cross_attn_img_to_chem(query=img_emb, key=chem_emb, value=chem_emb)
        chem_attn, _ = self.cross_attn_chem_to_img(query=chem_emb, key=img_emb, value=img_emb)

        fused = img_attn + chem_attn
        fused = self.layernorm(fused.squeeze(1))
        fused = self.dropout(F.relu(fused))
        return fused

class MultiModalModel(nn.Module):
    def __init__(self, 
                 img_dim, 
                 chem_model_name, 
                 fused_dim, 
                 hidden_dim, 
                 num_heads=4,
                 num_self_attention_layers=2,
                 num_classes=None, 
                 task='classification',
                 freeze_chem_encoder=True,
                 dropout_prob=0.5  # New tunable dropout parameter
                 ):
        super().__init__()
        self.smiles_encoder = SMILESEncoder(chem_model_name, freeze=freeze_chem_encoder)
        chem_dim = self.smiles_encoder.model.config.hidden_size

        # Simplified imaging branch with dropout using dropout_prob
        self.img_projection = nn.Sequential(
            nn.Linear(img_dim, fused_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(fused_dim, fused_dim)
        )

        self.chem_projection = nn.Sequential(
            nn.Linear(chem_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),  # Normalize chemical features
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )

        self.chem_scale = nn.Parameter(torch.tensor(1.0))
        self.img_scale = nn.Parameter(torch.tensor(1.0))
    
        self.cross_attention = CrossAttentionFusion(hidden_dim=fused_dim, num_heads=num_heads)

        self.img_self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fused_dim, nhead=num_heads), num_layers=num_self_attention_layers
        )
        self.chem_self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fused_dim, nhead=num_heads), num_layers=num_self_attention_layers
        )

        # Prediction head using dropout_prob as well
        if task == 'classification':
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 1)
            )

        self.task = task

    def forward(self, img_emb, smiles, mode="both"):
        if mode == "img_only":
            img_emb = self.img_projection(img_emb)
            img_emb = self.img_self_attention(img_emb.unsqueeze(1)).squeeze(1)            
            fused_emb = img_emb
            chem_emb = None  # Unused modality
        elif mode == "chem_only":
            chem_emb = self.smiles_encoder(smiles)
            chem_emb = self.chem_projection(chem_emb)
            chem_emb = self.chem_self_attention(chem_emb.unsqueeze(1)).squeeze(1)
            fused_emb = chem_emb
            img_emb = None  # Unused modality
        else:  # mode == "both"
            chem_emb = self.smiles_encoder(smiles)
            img_emb = self.img_projection(img_emb)
            chem_emb = self.chem_projection(chem_emb)
            img_emb = self.img_self_attention(img_emb.unsqueeze(1)).squeeze(1)
            chem_emb = self.chem_self_attention(chem_emb.unsqueeze(1)).squeeze(1)
            fused_emb = self.cross_attention(img_emb, chem_emb)
        
        prediction = self.head(fused_emb)
        return prediction, img_emb, chem_emb
