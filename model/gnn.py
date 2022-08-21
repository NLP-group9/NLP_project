"""GNN Encoder and Transformer Decoder."""
from collections import defaultdict

import torch.nn as nn
from layer import GraphTransformer
from torch import Tensor
from torch_geometric.data import Data


class GNNEncoder(nn.Module):
    """A Transformer GNN with edge features followed by linear layers."""

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        dim_emb: int,
        num_conv: int,
        num_linear: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        """Initialise class.

        Args:
            embedding_layer: embedding layer to encode tokens
            dim_emb: dimension of embeddings
            num_conv: number of multi-head attention layers.
            num_linear: number of linear layers in feed forward layer.
            num_heads: number of heads.
            dropout_rate: probability for dropout layers
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.node_embeddings = embedding_layer
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


class TransformerDecoder(nn.Module):
    """Transformer Based Decoder."""

    def __init__(
        self,
        vocab: defaultdict,
        dim_emb: int,
        num_layers: int,
        heads: int,
        embedding_layer: nn.Embedding,
    ):
        """Initialization.

        Args:
            vocab: vocabulary.
            dim_emb: embedding dimension.
            num_layers: number of decoder layers.
            heads: number of heads in decoder.
            embedding_layer: embedding layer to encode tokens.
        """
        super().__init__()
        self.vocab = vocab
        self.embedding_layer = embedding_layer
        self.dim_emb = dim_emb
        self.num_layers = num_layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.dim_emb, nhead=heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.predict = nn.Linear(dim_emb, len(self.vocab))

    def forward(
        self,
        emb: Tensor,
        data: Data,
    ) -> Tensor:
        """Forward function of decoder layers.

        Args:
            emb: graph embeddings of GNN encoder.
            data: Data object to process.

        Returns:
            logits: decoded tensor.
        """
        # reshape
        emb = emb.reshape(data.batch.size(0), -1, emb.shape[-1])  # (bs, num_nodes, dim_emb)
        emb = emb.permute(1, 0, 2)  # (num_nodes, bs, dim_emb)
        tgt_emb = self.embedding_layer(data.y).reshape(
            data.batch.size(0), -1, emb.shape[-1]
        )  # (bs, num_nodes, dim_emb)

        doc_pad_mask = data.doc_padding_masks.reshape(data.batch.size(0), -1)  # (bs, num_nodes)
        tgt_pad_mask = data.summ_key_padding_masks.reshape(
            data.batch.size(0), -1
        )  # (bs, num_nodes)

        output = self.transformer_decoder(
            tgt_emb, emb, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=doc_pad_mask
        )
        output = output.permute(1, 0, 2)  # (bs, num_nodes, dim_emb)
        logits = self.log_softmax(self.predict(output))  # (bs, num_nodes, len(vocab))
        return logits


class EncoderDecoder(nn.Module):
    """Encoder and Decoder."""

    def __init__(
        self,
        vocab: defaultdict,
        dim_emb: int,
        num_conv: int,
        num_linear: int,
        num_heads_decoder: int,
        num_decoder_layer: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        """Initialization.

        Args:
            vocab: vocabulary.
            dim_emb: embedding dimension.
            num_conv: number of multi-head attention in encoder.
            num_linear: number of linear layer in encoder.
            num_heads_decoder: number of heads in transformer in decoder.
            num_decoder_layer: number of transformer layer in decoder.
            num_heads: number of heads in encoder
            dropout_rate: dropout rate.
        """
        super().__init__()
        self.vocab = vocab
        self.dim_emb = dim_emb
        self.num_conv = num_conv
        self.num_linear = num_linear
        self.num_heads_decoder = num_heads_decoder
        self.num_decoder_layer = num_decoder_layer
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.dim_emb)
        self.encoder = GNNEncoder(
            dim_emb=self.dim_emb,
            num_conv=self.num_conv,
            num_linear=self.num_linear,
            dropout_rate=self.dropout_rate,
            embedding_layer=self.embedding_layer,
        )
        self.decoder = TransformerDecoder(
            vocab=self.vocab,
            dim_emb=self.dim_emb,
            num_layers=self.num_decoder_layer,
            heads=self.num_heads_decoder,
            embedding_layer=self.embedding_layer,
        )

    def forward(self, data: Data) -> Tensor:
        """Forward function for encoder decoder.

        Args:
            data: Data object to process.

        Returns:
            logits: decoded tensor.
        """
        emb = self.encoder(data)
        logits = self.decoder(emb, data)
        return logits
