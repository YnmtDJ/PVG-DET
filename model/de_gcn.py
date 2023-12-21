import torch
from torch import nn
import torch.nn.functional as F

from model.vig.vig import ViG


class DeGCN(nn.Module):
    """
    End-to-end object detection with graph convolution network and transformer.
    """
    def __init__(self, num_classes, num_queries=100, d_model=192):
        """
        :param num_classes: The number of object classes.
        :param num_queries: The number of object queries, ie detection slot. This is the maximal number of objects can
                            detect in a single image. For COCO, we recommend 100 queries.
        :param d_model: The hidden dimension.
        """
        super(DeGCN, self).__init__()
        self.vig = ViG(3, d_model)
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, 4*d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6, nn.LayerNorm(d_model))
        self.class_embed = nn.Linear(d_model, num_classes+1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, inputs):
        """
        :param inputs: The input images.
        :return: It returns a dict with the following elements:
                "pred_logits": The classification logits (including no-object) for all queries.
                "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, width, height). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the non-normalization bounding box. TODO
        """
        x = self.vig(inputs)  # (batch_size, d_model, height/16, width/16)
        batch_size, d_model, _, _ = x.shape
        x = x.reshape(batch_size, d_model, -1).permute(0, 2, 1)  # (batch_size, num_points, d_model)
        outputs = self.decoder(self.query_embed.repeat(batch_size, 1, 1), x)  # (batch_size, num_queries, d_model)
        outputs_class = self.class_embed(outputs)  # (batch_size, num_queries, num_classes)
        outputs_coord = self.bbox_embed(outputs).sigmoid()  # (batch_size, num_queries, 4)
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        :param input_dim: The input dimension.
        :param hidden_dim: The hidden dimension.
        :param output_dim: The output dimension.
        :param num_layers: The number of layers.
        """
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
