import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou, box_convert


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR. (Set Prediction)

    The process happens in two steps:
      1) we compute hungarian assignment between ground truth boxes and the outputs of the model
      2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict=None, no_object_coef=0.1):
        """
        Create the criterion.
        :param num_classes: Number of object categories, omitting the special no-object category
        :param weight_dict: Dict containing as key the names of the losses and as values their relative weight.
                            (default: {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2})
        :param no_object_coef: Relative classification weight applied to the no-object category (object category is 1)
        """
        super(SetCriterion, self).__init__()
        if weight_dict is None:
            weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
            self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = no_object_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        :param outputs: Dict of tensors, see the output specification of the model for the format
        :param targets: List of dicts, such that len(targets) == batch_size.
                        The expected keys in each dict depends on the losses applied, see each loss' doc.
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        # TODO distribute computation
        num_boxes = sum(len(target["labels"]) for target in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        loss = sum(self.weight_dict[k]*losses[k] for k, v in losses.items() if k in self.weight_dict)
        return loss, losses

    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss (NLL)
        Targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        device = outputs['pred_logits'].device
        src_logits = outputs['pred_logits']  # (batch_size, num_queries, num_classes+1)

        target_classes_o = torch.cat([target["labels"][index_j] for target, (_, index_j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        idx = self._get_src_permutation_idx(indices)  # (batch_idx, src_idx) Fancy indexing
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.permute(0, 2, 1), target_classes, self.empty_weight.to(device))
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, width, height).
        """
        assert 'pred_boxes' in outputs
        device = outputs['pred_boxes'].device
        idx = self._get_src_permutation_idx(indices)  # (batch_idx, src_idx) Fancy indexing
        src_boxes = outputs['pred_boxes'][idx]  # (num_target_boxes, 4)
        target_boxes = []
        for target, (_, index_j) in zip(targets, indices):  # notice that the target boxes must be normalized
            height, width = target["boxes"].canvas_size
            target_boxes.append(target["boxes"][index_j]/torch.tensor([width, height, width, height], device=device))
        target_boxes = torch.cat(target_boxes, dim=0)  # (num_target_boxes, 4)
        num_boxes = target_boxes.shape[0]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_convert(src_boxes, 'cxcywh', 'xyxy'),
            box_convert(target_boxes, 'cxcywh', 'xyxy')
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices (Fancy indexing)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        """
        Creates the matcher
        :param cost_class: The relative weight of the classification error in the matching cost
        :param cost_bbox: The relative weight of the L1 error of the bounding box coordinates in the matching cost
        :param cost_giou: The relative weight of the GIoU loss of the bounding box in the matching cost
        """
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class >= 0 and cost_bbox >= 0 and cost_giou >= 0, "any costs can't be negative"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching
        :param outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logit
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
        :param targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        :return: A list of size batch_size, containing tuples of (index_i, index_j) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        device = outputs["pred_logits"].device
        batch_size, num_queries, _ = outputs["pred_logits"].shape

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (batch_size * num_queries, num_classes)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (batch_size * num_queries, 4)

        # Also concat the target labels and boxes (normalized by the image size)
        tgt_ids = torch.cat([target["labels"] for target in targets])
        tgt_bbox = []
        for target in targets:
            height, width = target["boxes"].canvas_size  # image size
            tgt_bbox.append(target["boxes"]/torch.tensor([width, height, width, height]).to(device))
        tgt_bbox = torch.cat(tgt_bbox)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # (batch_size * num_queries, num_target_boxes)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (batch_size * num_queries, num_target_boxes)

        # Compute the giou cost between boxes
        # (batch_size * num_queries, num_target_boxes)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, 'cxcywh', 'xyxy'),
                                         box_convert(tgt_bbox, 'cxcywh', 'xyxy'))

        # Final cost matrix
        # (batch_size * num_queries, num_target_boxes)
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.reshape(batch_size, num_queries, -1).cpu()  # (batch_size, num_queries, num_target_boxes)

        # Split the cost matrix to get the result of match in each image
        sizes = [len(target["boxes"]) for target in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, dim=-1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
