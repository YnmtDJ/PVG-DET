"""
This is the file that contains the evaluation code for VisDrone dataset.
Mostly copied from https://github.com/tjiiv-cprg/visdrone-det-toolkit-python
"""
import numpy as np


def eval_det(all_gt, all_det, all_height, all_width, per_class=False):
    """
    :param all_gt: list of np.array[m, 8],  the boxes must be in 'xywh' format
    :param all_det: list of np.array[m, 6], truncation and occlusion not necessary
    :param all_height:
    :param all_width:
    :param per_class: if True, return the evaluation results for each class
    """
    all_gt_ = []
    all_det_ = []
    num_imgs = len(all_gt)
    for gt, det, height, width in zip(all_gt, all_det, all_height, all_width):
        gt, det = drop_objects_in_igr(gt, det, height, width)
        gt[:, 4] = 1 - gt[:, 4]  # set ignore flag
        all_gt_.append(gt)
        all_det_.append(det)
    return calc_accuracy(num_imgs, all_gt_, all_det_, per_class)


def drop_objects_in_igr(gt, det, img_height, img_width):
    gt_ignore_mask = gt[:, 5] == 0
    curgt = gt[np.logical_not(gt_ignore_mask)]
    igr_region = gt[gt_ignore_mask, :4].clip(min=1)
    if len(igr_region):
        igr_map = np.zeros((img_height, img_width), dtype=np.int32)

        for igr in igr_region:
            x1 = igr[0]
            y1 = igr[1]
            x2 = min(x1 + igr[2], img_width)
            y2 = min(y1 + igr[3], img_height)
            igr_map[y1 - 1:y2, x1 - 1:x2] = 1
        int_igr_map = create_int_img(igr_map)
        idx_left_gt = []

        for i, gtbox in enumerate(curgt):
            pos = np.round(gtbox[:4]).astype(np.int32).clip(min=1)
            x = max(1, min(img_width - 1, pos[0]))
            y = max(1, min(img_height - 1, pos[1]))
            w = pos[2]
            h = pos[3]
            tl = int_igr_map[y - 1, x - 1]
            tr = int_igr_map[y - 1, min(img_width, x + w) - 1]
            bl = int_igr_map[max(1, min(img_height, y + h)) - 1, x - 1]
            br = int_igr_map[max(1, min(img_height, y + h)) - 1,
                             min(img_width, x + w) - 1]
            igr_val = tl + br - tr - bl
            if igr_val / (h * w) < 0.5:
                idx_left_gt.append(i)

        curgt = curgt[idx_left_gt]

        idx_left_det = []
        for i, dtbox in enumerate(det):
            pos = np.round(dtbox[:4]).astype(np.int32).clip(min=1)
            x = max(1, min(img_width - 1, pos[0]))
            y = max(1, min(img_height - 1, pos[1]))
            w = pos[2]
            h = pos[3]
            tl = int_igr_map[y - 1, x - 1]
            tr = int_igr_map[y - 1, min(img_width, x + w) - 1]
            bl = int_igr_map[max(1, min(img_height, y + h)) - 1, x - 1]
            br = int_igr_map[max(1, min(img_height, y + h)) - 1,
                             min(img_width, x + w) - 1]
            igr_val = tl + br - tr - bl
            if igr_val / (h * w) < 0.5:
                idx_left_det.append(i)

        det = det[idx_left_det]

    return curgt, det


def create_int_img(img):
    int_img = np.cumsum(img, axis=0)
    np.cumsum(int_img, axis=1, out=int_img)
    return int_img


def calc_accuracy(num_imgs, all_gt, all_det, per_class=False):
    """
    :param num_imgs: int
    :param all_gt: list of np.array[m, 8], [:, 4] == 1 indicates ignored regions,
                    which should be dropped before calling this function
    :param all_det: list of np.array[m, 6], truncation and occlusion not necessary
    :param per_class:
    """
    assert num_imgs == len(all_gt) == len(all_det)

    ap = np.zeros((10, 10), dtype=np.float32)
    ar = np.zeros((10, 10, 4), dtype=np.float32)
    eval_class = []

    print('')
    for id_class in range(1, 11):
        print('evaluating object category {}/10...'.format(id_class))

        for gt in all_gt:
            if np.any(gt[:, 5] == id_class):
                eval_class.append(id_class - 1)

        x = 0
        for thr in np.linspace(0.5, 0.95, num=10):
            y = 0
            for max_dets in (1, 10, 100, 500):
                gt_match = []
                det_match = []
                for gt, det in zip(all_gt, all_det):
                    det_limited = det[:min(len(det), max_dets)]
                    mask_gt_cur_class = gt[:, 5] == id_class
                    mask_det_cur_class = det_limited[:, 5] == id_class
                    gt0 = gt[mask_gt_cur_class, :5]
                    dt0 = det_limited[mask_det_cur_class, :5]
                    gt1, dt1 = eval_res(gt0, dt0, thr)
                    # 1: matched, 0: unmatched, -1: ignore
                    gt_match.append(gt1[:, 4])
                    # [score, match type]
                    # 1: matched to gt, 0: unmatched, -1: matched to ignore
                    det_match.append(dt1[:, 4:6])
                gt_match = np.concatenate(gt_match, axis=0)
                det_match = np.concatenate(det_match, axis=0)

                idrank = det_match[:, 0].argsort()[::-1]
                tp = np.cumsum(det_match[idrank, 1] == 1)
                rec = tp / max(1, len(gt_match))  # including ignore (already dropped)
                if len(rec):
                    ar[id_class - 1, x, y] = np.max(rec) * 100

                y += 1

            fp = np.cumsum(det_match[idrank, 1] == 0)
            prec = tp / (fp + tp).clip(min=1)
            ap[id_class - 1, x] = voc_ap(rec, prec) * 100

            x += 1

    ap_all = np.mean(ap[eval_class, :])
    ap_50 = np.mean(ap[eval_class, 0])
    ap_75 = np.mean(ap[eval_class, 5])
    ar_1 = np.mean(ar[eval_class, :, 0])
    ar_10 = np.mean(ar[eval_class, :, 1])
    ar_100 = np.mean(ar[eval_class, :, 2])
    ar_500 = np.mean(ar[eval_class, :, 3])

    results = (ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500)

    if per_class:
        ap_classwise = np.mean(ap, axis=1)
        results += (ap_classwise,)

    print('Evaluation completed. The performance of the detector is presented as follows.')
    print('Average Precision  (AP) @[ IoU=0.50:0.95 | maxDets=500 ] = {}%.'.format(ap_all))
    print('Average Precision  (AP) @[ IoU=0.50      | maxDets=500 ] = {}%.'.format(ap_50))
    print('Average Precision  (AP) @[ IoU=0.75      | maxDets=500 ] = {}%.'.format(ap_75))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=  1 ] = {}%.'.format(ar_1))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets= 10 ] = {}%.'.format(ar_10))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=100 ] = {}%.'.format(ar_100))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=500 ] = {}%.'.format(ar_500))

    return results


def eval_res(gt0, dt0, thr):
    """
    :param gt0: np.array[ng, 5], ground truth results [x, y, w, h, ignore]
    :param dt0: np.array[nd, 5], detection results [x, y, w, h, score]
    :param thr: float, IoU threshold
    :return gt1: np.array[ng, 5], gt match types
             dt1: np.array[nd, 6], dt match types
    """
    nd = len(dt0)
    ng = len(gt0)

    # sort
    dt = dt0[dt0[:, 4].argsort()[::-1]]
    gt_ignore_mask = gt0[:, 4] == 1
    gt = gt0[np.logical_not(gt_ignore_mask)]
    ig = gt0[gt_ignore_mask]
    ig[:, 4] = -ig[:, 4]  # -1 indicates ignore

    dt_format = dt[:, :4].copy()
    gt_format = gt[:, :4].copy()
    ig_format = ig[:, :4].copy()
    dt_format[:, 2:] += dt_format[:, :2]  # [x2, y2] = [w, h] + [x1, y1]
    gt_format[:, 2:] += gt_format[:, :2]
    ig_format[:, 2:] += ig_format[:, :2]

    iou_dtgt = bbox_overlaps(dt_format, gt_format, mode='iou')
    iof_dtig = bbox_overlaps(dt_format, gt_format, mode='iof')
    oa = np.concatenate((iou_dtgt, iof_dtig), axis=1)

    # [nd, 6]
    dt1 = np.concatenate((dt, np.zeros((nd, 1), dtype=dt.dtype)), axis=1)
    # [ng, 5]
    gt1 = np.concatenate((gt, ig), axis=0)

    for d in range(nd):
        bst_oa = thr
        bstg = -1  # index of matched gt
        bstm = 0  # best match type
        for g in range(ng):
            m = gt1[g, 4]
            # if gt already matched, continue to next gt
            if m == 1:
                continue
            # if dt already matched, and on ignore gt, nothing more to do
            if bstm != 0 and m == -1:
                break
            # continue to next gt until better match is found
            if oa[d, g] < bst_oa:
                continue
            bst_oa = oa[d, g]
            bstg = g
            bstm = 1 if m == 0 else -1  # 1: matched to gt, -1: matched to ignore

        # store match type for dt
        dt1[d, 5] = bstm
        # store match flag for gt
        if bstm == 1:
            gt1[bstg, 4] = 1

    return gt1, dt1


def voc_ap(rec, prec):
    mrec = np.concatenate(([0], rec, [1]))
    mpre = np.concatenate(([0], prec, [0]))
    for i in reversed(range(0, len(mpre)-1)):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
        eps(float):

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
