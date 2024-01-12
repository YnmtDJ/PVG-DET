import torch
from torch import nn

from model.vig.vig import Stem16
from model.common import MLP
from model.detr.transformer import TransformerDecoderLayer, TransformerDecoder
from model.position_embedding import PositionEmbedding2d
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
        self.stem = Stem16(3, d_model)
        self.pos_embed = PositionEmbedding2d()
        self.vig = ViG(d_model)
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
        decoder_layer = TransformerDecoderLayer(d_model, 8, 4*d_model)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, 6, decoder_norm, d_model=d_model)
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
                            See PostProcess for information on how to retrieve the non-normalization bounding box.
        """
        # backbone and position embedding
        x = self.stem(inputs)  # (batch_size, d_model, height/16, width/16)
        pos = self.pos_embed(x)

        # encoder
        memory = self.vig(x + pos)  # (batch_size, d_model, height/16, width/16)
        batch_size, d_model, _, _ = memory.shape
        memory = memory.reshape(batch_size, d_model, -1).permute(2, 0, 1)  # (num_points, batch_size, d_model)
        pos = pos.reshape(batch_size, d_model, -1).permute(2, 0, 1)

        # decoder
        query_embed = self.query_embed.unsqueeze(1).expand(-1, batch_size, -1)  # (num_queries, batch_size, d_model)
        tgt = torch.zeros_like(query_embed)
        outputs = self.decoder(tgt, memory, pos=pos, query_pos=query_embed)
        outputs = outputs.permute(1, 0, 2)  # (batch_size, num_queries, d_model)

        # classification and positioning
        outputs_class = self.class_embed(outputs)  # (batch_size, num_queries, num_classes)
        outputs_coord = self.bbox_embed(outputs).sigmoid()  # (batch_size, num_queries, 4)
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
