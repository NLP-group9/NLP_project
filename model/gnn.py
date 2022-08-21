"""GNN Encoder and Transformer Decoder."""
from collections import defaultdict

import torch.nn as nn
from layer import GraphTransformer
from torch import Tensor
from torch.nn import Embedding
from torch_geometric.data import Data


class GNNEncoder(nn.Module):
    """A Transformer GNN with edge features followed by linear layers."""

    def __init__(
        self,
        vocab: defaultdict,
        dim_emb: int,
        num_conv: int,
        num_linear: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        """Initialise class.

        Args:
            vocab: vocabulary of tokens.
            dim_emb: dimension of embeddings
            num_conv: number of multi-head attention layers.
            num_linear: number of linear layers in feed forward layer.
            num_heads: number of heads.
            dropout_rate: probability for dropout layers
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.node_embeddings = Embedding(num_embeddings=len(vocab), embedding_dim=dim_emb)
        self.num_conv = num_conv
        self.graph_transformers = nn.ModuleList(
            [
                GraphTransformer(
                    dim_emb=dim_emb, heads=num_heads, dropout=dropout_rate, num_linear=num_linear
                )
                for _ in range(self.num_conv)
            ]
        )

    def forward(self, data: Data) -> Tensor:
        """Forward function of GNN Encoder.

        Args:
            data: Data object to process.
        """
        # 1. embedding layer
        emb = self.node_embeddings(data.x)
        # 2. Attention blocks
        for i in range(self.num_con):
            emb = self.graph_transformers[i](emb, data.edge_index)
        return emb
