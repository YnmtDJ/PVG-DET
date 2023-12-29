import torch
from torch import nn


class PositionEmbedding2d(nn.Module):
    """
    2D Position Embedding using Sine and Cosine functions.
    """
    def __init__(self):
        super(PositionEmbedding2d, self).__init__()

    def forward(self, x):
        batch_size, d_model, height, width = x.shape
        device = x.device
        assert d_model % 4 == 0
        embed_h = self.get_1d_pos_embedding(d_model // 2, height).to(device)  # (d_model/2, height)
        embed_w = self.get_1d_pos_embedding(d_model // 2, width).to(device)  # (d_model/2, width)
        # (batch_size, d_model/2, height, width)
        embed_h = embed_h.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, width)
        embed_w = embed_w.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, height, 1)
        pos_embedding = torch.cat([embed_h, embed_w], dim=1)  # (batch_size, d_model, height, width)
        return pos_embedding

    def get_1d_pos_embedding(self, d_model, seq_len):
        """
        :param d_model: The hidden dimension.
        :param seq_len: The length of the sequence.
        """
        pos = torch.arange(seq_len, dtype=torch.float32)
        i = torch.arange(d_model // 2, dtype=torch.float32)
        freq = 1 / (10000 ** (2 * i / d_model))
        position = torch.einsum("i,j->ij", freq, pos)  # (d_model/2, seq_len)
        pos_embedding = torch.zeros([d_model, seq_len])  # (d_model, seq_len)
        pos_embedding[0::2, :] = torch.sin(position)
        pos_embedding[1::2, :] = torch.cos(position)
        return pos_embedding
