import torch
from torch import nn


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
      1) we compute hungarian assignment between ground truth boxes and the outputs of the model
      2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self):
        pass

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        :param outputs: Dict of tensors, see the output specification of the model for the format.
        :param targets: List of dicts, such that len(targets) == batch_size.
        """




class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcher
        :param cost_class: The relative weight of the classification error in the matching cost
        :param cost_bbox: The relative weight of the L1 error of the bounding box coordinates in the matching cost
        :param cost_giou: The relative weight of the GIoU loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class > 0 and cost_bbox > 0 and cost_giou > 0, "any costs can't be negative"

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
        batch_size, num_queries, _ = outputs["pred_logits"].shape

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([target["labels"] for target in targets])
        tgt_bbox = torch.cat([target["boxes"] for target in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
