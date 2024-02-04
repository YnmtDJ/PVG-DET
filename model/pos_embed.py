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

        pos_embedding = get_2d_pos_embedding(d_model, height, width, device)  # (d_model, height, width)
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, d_model, height, width)
        return pos_embedding


class RelativePositionEmbedding2d(nn.Module):
    """
    2D Relative Position Embedding using Sine and Cosine functions.
    """
    def __init__(self):
        super(RelativePositionEmbedding2d, self).__init__()

    def forward(self, x):
        batch_size, d_model, height, width = x.shape
        device = x.device
        assert d_model % 4 == 0

        # calculate 2d position embedding
        pos_embedding = get_2d_pos_embedding(d_model, height, width, device)  # (d_model, height, width)
        pos_embedding = pos_embedding.reshape(d_model, -1)  # (d_model, num_points)

        # calculate relative position embedding
        # (num_points, num_points)
        relative_pos = 2 * torch.matmul(pos_embedding.transpose(0, 1), pos_embedding) / d_model
        relative_pos = relative_pos.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_points, num_points)
        return relative_pos


def get_1d_pos_embedding(d_model, seq_len, device):
    """
    :param d_model: The hidden dimension.
    :param seq_len: The length of the sequence.
    :param device: The desired device of returned tensor.
    """
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    i = torch.arange(d_model // 2, dtype=torch.float32, device=device)
    freq = 1 / (10000 ** (2 * i / d_model))
    position = torch.einsum("i,j->ij", freq, pos)  # (d_model/2, seq_len)
    pos_embedding = torch.zeros([d_model, seq_len], device=device)  # (d_model, seq_len)
    pos_embedding[0::2, :] = torch.sin(position)
    pos_embedding[1::2, :] = torch.cos(position)
    return pos_embedding


def get_2d_pos_embedding(d_model, height, width, device):
    """
    :param d_model: The hidden dimension.
    :param height: The height of the image.
    :param width: The width of the image.
    :param device: The desired device of returned tensor.
    """
    embed_h = get_1d_pos_embedding(d_model // 2, height, device)  # (d_model/2, height)
    embed_w = get_1d_pos_embedding(d_model // 2, width, device)  # (d_model/2, width)
    # (d_model/2, height, width)
    embed_h = embed_h.unsqueeze(-1).expand(-1, -1, width)
    embed_w = embed_w.unsqueeze(1).expand(-1, height, -1)
    pos_embedding = torch.cat([embed_h, embed_w], dim=0)  # (d_model, height, width)
    return pos_embedding
