"""Define layers for transformer attention block."""
from torch import Tensor, nn
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.conv import TransformerConv


class MLP(nn.Module):
    """Feed forward layer."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        inter_dim: int,
        layers: int,
        activation: str = "ReLU",
        dropout: float = 0.0,
    ):
        """Initialization.

        Args:
            dim_in: dimension of input embeddings
            inter_dim: intermediate dimension of embeddings
            dim_out: output dimension
            activation: activation type
            dropout: dropout rate of MLP
            layers: number of linear layers.
        """
        super().__init__()
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        if activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        self.dropout = dropout
        assert self.dropout > 0
        self.layers = layers
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.inter_dim = inter_dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x: vector need to be processed.

        Returns:
            a build-up MLP.
        """
        layers = []
        dim_in = [self.dim_in] + [self.inter_dim for _ in range(self.layers - 1)]
        dim_out = [self.inter_dim for _ in range(self.layers - 1)] + [self.dim_out]
        for i in range(self.layers):
            layers.append(nn.Linear(dim_in[i], dim_out[i]))
            if i != self.layer - 1:
                layers.append(self.activation)
                layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)(x)


class GraphTransformer(nn.Module):
    """Graph Transformer, multi-head attention combined with feed forward layer."""

    def __init__(
        self,
        dim_emb: int,
        heads: int,
        dropout: float,
        num_linear: int,
    ):
        """Initialization of GraphTransformer.

        Args:
            dim_emb: input embedding dimensions.
            heads: number of heads in multi-head attention.
            dropout: dropout rate.
            num_linear: number of feed forward layers.
        """
        super().__init__()
        self.dim_emb = dim_emb
        self.in_channels = dim_emb
        self.out_channels = self.in_channels // dim_emb
        assert self.out_channels * dim_emb == self.in_channels
        self.heads = (heads,)
        self.dropout = dropout
        self.num_linear = num_linear
        self.attention_layer = TransformerConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            heads=heads,
            dropout=self.dropout,
        )
        self.mlp = MLP(
            dim_in=self.out_channels,
            dim_out=self.out_channels,
            inter_dim=4 * self.out_channels,
            layers=num_linear,
        )
        self.norm = LayerNorm(self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of GraphTransformer.

        Args:
            x: input embeddings.
        """
        # multi-head attention layer
        x_atten = self.attention_layer(x)
        # residual connection
        x = self.norm(x + x_atten)
        # feed forward layer
        x_mlp = self.mlp(x)
        # residual connection
        x = self.norm(x + x_mlp)
        return x
