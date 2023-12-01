import torch
from torch import nn
import torch.nn.functional as F

from model.vig.vig import ViG


class DeGCN(nn.Module):
    """
    End-to-end object detection with graph convolution network and transformer.
    """
    def __init__(self, max_id, num_queries=100, d_model=192):
        super(DeGCN, self).__init__()
        self.vig = ViG()
        self.query_embed = nn.Parameter(torch.randn(num_queries, 1, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, 4*d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6, nn.LayerNorm(d_model))
        self.class_embed = nn.Linear(d_model, max_id + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, inputs):
        x = self.vig(inputs)  # (batch_size, num_dims, height/16, width/16)
        batch_size, num_dims, _, _ = x.shape
        x = x.reshape(batch_size, num_dims, -1).permute(2, 0, 1)  # (num_points, batch_size, num_dims)
        outputs = self.decoder(self.query_embed.repeat(1, batch_size, 1), x)
        outputs_class = self.class_embed(outputs)
        outputs_coord = self.bbox_embed(outputs).sigmoid()
        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.Sequential(*[
            nn.Linear(in_ch, out_ch) for in_ch, out_ch in zip([input_dim] + h_dims, h_dims + [output_dim])
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




