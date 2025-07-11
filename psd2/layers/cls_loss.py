import torch
import torch.nn as nn
import torch.nn.functional as F
from psd2.config import configurable
from fvcore.nn import sigmoid_focal_loss_jit

# TODO distributed average
class ReidCELoss(nn.Module):
    @configurable()
    def __init__(self, num_lb, feat_len, loss_weight):
        super().__init__()
        self.lb_layer = nn.Linear(feat_len, num_lb)
        self.loss_weight = loss_weight
        nn.init.normal_(self.lb_layer.weight, std=0.01)
        nn.init.constant_(self.lb_layer.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.PERSON_SEARCH.REID.LOSS
        assert hasattr(loss_cfg, "CE")
        ce_cfg = loss_cfg.CE
        ret = {
            "num_lb": ce_cfg.NUM_LB,
            "feat_len": ce_cfg.FEAT_DIM,
            "loss_weight": ce_cfg.LOSS_WEIGHT,
        }
        return ret

    def forward(self, pfeats, pids, *args, **kws):
        pos_mask = pids > -1
        if pos_mask.sum().item() == 0:
            return {
                "loss_reid_ce": torch.zeros(1, dtype=pfeats.dtype, device=pfeats.device)
            }
        pfeats = pfeats[pos_mask]
        logits = self.lb_layer(pfeats)
        loss_v = F.cross_entropy(logits, pids[pos_mask], reduction="mean")
        return {"loss_reid_ce": loss_v * self.loss_weight}


class ReidBCELoss(nn.Module):
    @configurable()
    def __init__(
        self,
        num_lb,
        feat_len,
        loss_weight,
        use_focal=False,
        focal_alpha=1,
        focal_gamma=2,
    ):
        super().__init__()
        self.lb_layer = nn.Linear(feat_len, num_lb)
        self.loss_weight = loss_weight
        nn.init.normal_(self.lb_layer.weight, std=0.01)
        nn.init.constant_(self.lb_layer.bias, 0)
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.PERSON_SEARCH.REID.LOSS
        assert hasattr(loss_cfg, "CE")
        ce_cfg = loss_cfg.CE
        ret = {
            "num_lb": ce_cfg.NUM_LB,
            "feat_len": ce_cfg.FEAT_DIM,
            "loss_weight": ce_cfg.LOSS_WEIGHT,
        }
        if ce_cfg.USE_FOCAL:
            ret["use_focal"] = True
            ret["focal_alpha"] = ce_cfg.FOCAL_ALPHA
            ret["focal_gamma"] = ce_cfg.FOCAL_GAMMA
        else:
            ret["use_focal"] = False
        return ret

    def forward(self, pfeats, pids, *args, **kws):
        loss_name = "loss_reid_bce" if not self.use_focal else "loss_reid_focal"
        pos_mask = pids > -1
        if pos_mask.sum().item() == 0:
            return {loss_name: torch.zeros(1, dtype=pfeats.dtype, device=pfeats.device)}
        pfeats = pfeats[pos_mask]
        pids = pids[pos_mask]
        targets = F.one_hot(pids, num_classes=self.num_classes)
        logits = self.lb_layer(pfeats)
        if self.use_focal:
            loss_v = sigmoid_focal_loss_jit(
                logits,
                targets.to(logits.dtype),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="mean",
            )
        else:
            loss_v = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="mean"
            )
        return {loss_name: loss_v * self.loss_weight}
