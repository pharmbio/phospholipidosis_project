import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn as nn

class SimpleMergeFusion(nn.Module):
    def __init__(self, method="concat"):
        """
        :param method: "mean", "sum", or "concat"
        """
        super().__init__()
        self.method = method.lower()
        
    def forward(self, img_emb, chem_emb):
        # Both have shape (batch_size, embed_dim)
        if self.method == "mean":
            return 0.5 * img_emb + 0.5 * chem_emb
        elif self.method == "sum":
            return img_emb + chem_emb
        elif self.method == "concat":
            # In this case, we return (batch_size, embed_dim * 2)
            return torch.cat([img_emb, chem_emb], dim=-1)
        else:
            raise ValueError(f"Unknown simple merge method: {self.method}")


### 🔹 Transformer Fusion for Multi-Modal Embeddings
class TransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ffn_dim=256, dropout=0.1, agg_mode="cls"):
        """
        Transformer Fusion module to aggregate multimodal embeddings.
        Supports 'cls', 'mean', or 'max' aggregation methods.

        Parameters:
        - embed_dim: Feature dimension of embeddings
        - num_layers: Number of Transformer layers
        - num_heads: Number of attention heads
        - ffn_dim: Feedforward layer dimension
        - dropout: Dropout probability
        - agg_mode: Aggregation strategy ("cls", "mean", "max")
        """
        super().__init__()
        self.agg_mode = agg_mode.lower()
        self.latent_dim = embed_dim

        self.embed_to_latent = nn.Linear(embed_dim, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ffn_dim, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.latent_to_embed = nn.Linear(self.latent_dim, embed_dim)

        # Optional CLS token for sequence aggregation
        if self.agg_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))

    def forward(self, fusion_seq, fusion_mask=None):
        """
        Applies Transformer Fusion on the input sequence.

        Parameters:
        - fusion_seq: (batch_size, seq_length, feature_dim)
        - fusion_mask: Optional mask for padded elements (batch_size, seq_length)

        Returns:
        - Aggregated feature representation
        """
        batch_size = fusion_seq.shape[0]

        fusion_seq = self.embed_to_latent(fusion_seq)
        if self.agg_mode == "cls":
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
            fusion_seq = torch.cat([cls_tokens, fusion_seq], dim=1)

        fusion_seq = self.transformer_encoder(fusion_seq, src_key_padding_mask=fusion_mask)
        fusion_seq = self.latent_to_embed(fusion_seq)

        if self.agg_mode == "cls":
            return fusion_seq[:, 0, :]
        elif self.agg_mode == "mean":
            return fusion_seq.mean(dim=1)
        elif self.agg_mode == "max":
            return fusion_seq.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid aggregation method: {self.agg_mode}")

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        """
        Two-way cross-attention:
        - Query=img_emb, Key/Value=chem_emb
        - Query=chem_emb, Key/Value=img_emb
        Then merges them with some residual/MLP if desired.
        """
        super().__init__()
        self.cross_attn_img_to_chem = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn_chem_to_img = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_emb, chem_emb):
        """
        :param img_emb: (batch, embed_dim)
        :param chem_emb: (batch, embed_dim)
        """
        # For multihead attention, we need a time/sequence dimension: (batch, seq=1, embed_dim)
        img_emb = img_emb.unsqueeze(1)  # => (batch, 1, embed_dim)
        chem_emb = chem_emb.unsqueeze(1)

        # Cross attention: from image to chem
        img_attn, _ = self.cross_attn_img_to_chem(query=img_emb, key=chem_emb, value=chem_emb)
        # Cross attention: from chem to image
        chem_attn, _ = self.cross_attn_chem_to_img(query=chem_emb, key=img_emb, value=img_emb)

        # Combine them
        fused = img_attn + chem_attn  # shape: (batch, 1, embed_dim)
        fused = self.layernorm(fused.squeeze(1))  # => (batch, embed_dim)
        fused = self.dropout(F.relu(fused))
        return fused


### 🔹 SMILES Encoder with Support for ChemBERTa, MoL-T5, and GPS+
class SMILESEncoder(nn.Module):
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", model_type="chemberta", freeze=True):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type in ["chemberta", "molt5"]:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hidden_dim = self.model.config.hidden_size
        elif self.model_type == "gps+":
            self.hidden_dim = 300  # GPS+ feature dimension
            self.model = self._init_gps_encoder()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if freeze and self.model_type in ["chemberta", "molt5"]:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, smiles):
        if self.model_type in ["chemberta", "molt5"]:
            inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: val.to(next(self.model.parameters()).device) for key, val in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
            return embeddings

        elif self.model_type == "gps+":
            graphs = [self.smiles_to_graph(s) for s in smiles]
            batch_graph = Data.from_dict({"batch": torch.tensor([i for i in range(len(graphs))])})
            batch_graph.x = torch.cat([g.x for g in graphs])
            batch_graph.edge_index = torch.cat([g.edge_index for g in graphs], dim=1)
            batch_graph.batch = torch.cat([torch.full((g.x.size(0),), i, dtype=torch.long) for i, g in enumerate(graphs)])

            return self.model(batch_graph)

    def _init_gps_encoder(self):
        conv = GINConv(nn.Linear(300, 300))
        return nn.Sequential(conv, nn.ReLU(), global_mean_pool)

    def smiles_to_graph(self, smiles):
        from rdkit import Chem
        from torch_geometric.utils import from_networkx
        import networkx as nx

        mol = Chem.MolFromSmiles(smiles)
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), x=torch.tensor(atom.GetAtomicNum()).unsqueeze(0))
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        return from_networkx(graph)


### 🔹 MultiModal Model with Transformer Fusion
class MultiModalModel(nn.Module):
    def __init__(self, 
                 img_dim, 
                 chem_model_name, 
                 chem_model_type,
                 fused_dim,
                 hidden_dim,
                 num_heads=4,
                 num_self_attention_layers=2,
                 num_classes=None, 
                 task='classification',
                 freeze_chem_encoder=True,
                 dropout_prob=0.5,
                 fusion_type="transformer",
                 fusion_agg="cls"):
        super().__init__()
        self.task = task
        # define an attribute to store the string
        self.fusion_type = fusion_type

        # 1) SMILES encoder
        self.smiles_encoder = SMILESEncoder(chem_model_name, chem_model_type, freeze=freeze_chem_encoder)
        chem_dim = self.smiles_encoder.hidden_dim

        # 2) Image projection
        self.img_projection = nn.Sequential(
            nn.Linear(img_dim, fused_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(fused_dim, fused_dim)
        )

        # 3) Chemical projection
        self.chem_projection = nn.Sequential(
            nn.Linear(chem_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )

        # 4) Fusion mechanism
        if fusion_type == "transformer":
            self.fusion_layer = TransformerFusion(
                embed_dim=fused_dim, 
                num_layers=num_self_attention_layers,
                num_heads=num_heads, 
                ffn_dim=hidden_dim, 
                dropout=dropout_prob,
                agg_mode=fusion_agg
            )
            final_dim = fused_dim
        elif fusion_type == "cross_attention":
            self.fusion_layer = CrossAttentionFusion(
                hidden_dim=fused_dim, 
                num_heads=num_heads, 
                dropout=dropout_prob
            )
            final_dim = fused_dim
        elif fusion_type == "simple":
            # or "simple_merge"
            self.fusion_layer = SimpleMergeFusion(method="concat")  # or "sum"/"mean"
            final_dim = fused_dim * 2  # if "concat"
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # 5) Prediction head
        if task == 'classification':
            self.head = nn.Sequential(
                nn.Linear(final_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, num_classes)
            )
        else:  # regression
            self.head = nn.Sequential(
                nn.Linear(final_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, img_emb, smiles, mode="both"):
        """
        :param img_emb: (batch_size, img_dim)
        :param smiles: list of SMILES strings (length = batch_size)
        :param mode: "img_only", "chem_only", or "both"
        """
        # 1) Chemical embeddings
        chem_emb = self.smiles_encoder(smiles)
        # 2) Projection
        img_emb = self.img_projection(img_emb)
        chem_emb = self.chem_projection(chem_emb)

        # 3) Depending on the mode
        if mode == "img_only":
            fused_emb = img_emb
        elif mode == "chem_only":
            fused_emb = chem_emb
        else:
            # BOTH
            if self.fusion_type == "transformer":
                # shape => (batch_size, 2, embed_dim)
                fusion_input = torch.stack([img_emb, chem_emb], dim=1)
                fused_emb = self.fusion_layer(fusion_input)
            elif self.fusion_type == "cross_attention":
                # cross attention wants separate embeddings
                fused_emb = self.fusion_layer(img_emb, chem_emb)
            elif self.fusion_type == "simple":
                # simple merging also wants separate embeddings
                fused_emb = self.fusion_layer(img_emb, chem_emb)
            else:
                raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        # 4) Final prediction
        prediction = self.head(fused_emb)
        return prediction, img_emb, chem_emb