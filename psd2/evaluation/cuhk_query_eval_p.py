from tqdm import tqdm
import torch
import logging

import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score

import logging
from .query_evaluator import QueryEvaluator
from torch.utils.data import Dataset, DataLoader
from psd2.structures import Boxes, BoxMode, pairwise_iou

logger = logging.getLogger(__name__)  # setup_logger()


class CuhkQueryEvaluatorP(QueryEvaluator):
    def __init__(self, *args, **kws) -> None:
        super().__init__(*args, **kws)
        self.inf_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs:
                a batch of
                {
                    "query":
                    {
                        "image": augmented image tensor
                        "instances": an Instances object with attrs
                            {
                                image_size: hw (int, int)
                                file_name: full path string
                                image_id: filename string
                                gt_boxes: Boxes (1 , 4)
                                gt_classes: tensor full with 0s
                                gt_pids: person identity tensor (1,)
                                org_img_size: hw before augmentation (int, int)
                                org_gt_boxes: Boxes before augmentation
                            }
                    }
                    "gallery": a list of Instances objects
                    [
                        Instances object with attr:
                            file_name,
                            image_id,
                            gt_boxes: (1,4) box of the true positive
                            gt_pids: (1,) query pid
                        ...
                    ]
                }
            outputs:
                a batch of instances with attrs
                {
                    reid_feats: tensor (1,pfeat_dim)
                }
                or
                a batch of empty instances
        """
        # NOTE save pairs (query gt instances,gallery gt instances, query pred instances)
        save_inputs = []
        for item_in, inst_pred in zip(inputs, outputs):
            q_gt_instances = item_in["query"]["instances"].to(self._cpu_device)
            g_gt_instances = [inst.to(self._cpu_device) for inst in item_in["gallery"]]
            q_pred_instances = inst_pred.to(self._cpu_device)
            save_inputs.append((q_gt_instances, g_gt_instances, q_pred_instances))
        self.inf_results.append(save_inputs)

    def evaluate(self):
        eval_dataset = EvaluatorDataset(self.inf_results, self)
        eval_worker = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            collate_fn=lambda x: x,
        )
        logger.info("Parallel evaluating on {}:".format(self.dataset_name))
        with tqdm(total=len(eval_worker)) as pbar:
            for b_rst in eval_worker:
                rst_aps, rst_accs = b_rst[0]
                for st in self.det_score_thresh:
                    self.aps[st].extend(rst_aps[st])
                    self.accs[st].extend(rst_accs[st])
                pbar.update(1)
        return super().evaluate()



class EvaluatorDataset(Dataset):
    def __init__(self, eval_inputs, eval_wref):
        self.eval_inputs = eval_inputs
        self.eval_ref = eval_wref

    def __len__(self):
        return len(self.eval_inputs)

    def __getitem__(self, idx):
        inputs = self.eval_inputs[idx]
        rst_aps = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_accs = {st: [] for st in self.eval_ref.det_score_thresh}
        for bi, in_dict in enumerate(inputs):
            q_gt_instances, g_gt_instances_list, q_pred_instances = in_dict
            q_imgid = q_gt_instances.image_id
            q_pid = q_gt_instances.gt_pids
            q_box = q_gt_instances.org_gt_boxes
            y_trues = {dst: [] for dst in self.eval_ref.det_score_thresh}
            y_scores = {dst: [] for dst in self.eval_ref.det_score_thresh}
            count_gts = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            count_tps = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            for dst in self.eval_ref.det_score_thresh:
                if q_pred_instances.has("reid_feats"):
                    feat_q = q_pred_instances.reid_feats
                else:
                    (
                        query_img_boxes_t,
                        query_img_feats,
                    ) = self.eval_ref._get_gallery_dets(q_imgid, dst)[
                        :, :4
                    ], self.eval_ref._get_gallery_feats(
                        q_imgid, dst
                    )
                    if query_img_boxes_t.shape[0] == 0:
                        # no detection in this query image
                        logger.warning(
                            "Undetected query person in {} !".format(q_imgid)
                        )
                        continue
                    else:
                        ious = pairwise_iou(
                            q_box, Boxes(query_img_boxes_t, BoxMode.XYXY_ABS)
                        )
                        max_iou, nmax = torch.max(ious, dim=1)
                        if max_iou.item() < 0.4:
                            logger.warning(
                                "Low-quality {} query person detected in {} !".format(
                                    max_iou.item(), q_imgid
                                )
                            )
                        feat_q = query_img_feats[nmax.item()]

                # feat_q = F.normalize(feat_q[None]).squeeze(0)  # NOTE keep post norm
                name2sim = {}

                # 1. Go through the gallery samples defined by the protocol
                for item in g_gt_instances_list:
                    gallery_imname = item.image_id
                    # some contain the query (gt not empty), some not
                    gt_boxes = item.org_gt_boxes
                    count_gts[dst] += len(gt_boxes) > 0
                    # compute distance between query and gallery dets
                    if gallery_imname not in self.eval_ref.infs:
                        continue
                    det, feat_g = self.eval_ref._get_gallery_dets(
                        gallery_imname, dst
                    ), self.eval_ref._get_gallery_feats(gallery_imname, dst)
                    # no detection in this gallery, skip it
                    if det.shape[0] == 0:
                        continue
                    # get L2-normalized feature matrix NxD
                    # feat_g = F.normalize(feat_g)  # NOTE keep post norm
                    # compute cosine similarities
                    sim = torch.mm(feat_g, feat_q.view(-1)[:, None]).squeeze(
                        1
                    )  # n x 1 -> n
                    if gallery_imname in name2sim:
                        continue
                    name2sim[gallery_imname] = sim

                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    if len(gt_boxes) > 0:
                        hw = gt_boxes.get_sizes().squeeze(0)
                        iou_thresh = min(
                            0.5, (hw[1] * hw[0] * 1.0) / ((hw[1] + 10) * (hw[0] + 10))
                        )
                        inds = torch.argsort(sim)
                        inds = inds.tolist()[::-1]
                        inds = torch.tensor(inds, dtype=torch.long)
                        sim = name2sim[gallery_imname][inds]
                        det = det[inds]
                        # only set the first matched det as true positive
                        for j, roi in enumerate(det[:, :4]):
                            if (
                                box_iou(roi[None, :], gt_boxes.tensor).squeeze().item()
                                >= iou_thresh
                            ):
                                label[j] = 1
                                count_tps[dst] += 1
                                break
                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())

                # 2. Compute AP for this probe (need to scale by recall rate)

                y_score = np.asarray(y_scores[dst])
                y_true = np.asarray(y_trues[dst])
                assert count_tps[dst] <= count_gts[dst]
                recall_rate = count_tps[dst] * 1.0 / count_gts[dst]
                ap = (
                    0
                    if count_tps[dst] == 0
                    else average_precision_score(y_true, y_score) * recall_rate
                )
                rst_aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true_o = y_true[inds]
                rst_accs[dst].append(
                    [min(1, sum(y_true_o[:k])) for k in self.eval_ref.topks]
                )

        return rst_aps, rst_accs
