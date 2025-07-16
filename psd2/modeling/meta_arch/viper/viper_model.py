import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple, Union
import logging
import math
from .base_tbps import SearchBaseTBPS
from ..build import META_ARCH_REGISTRY
from psd2.utils.events import get_event_storage
from psd2.config.config import configurable
from psd2.modeling.roi_heads.roi_heads import (
    Res5ROIHeads,
    add_ground_truth_to_proposals,
)
from psd2.modeling.box_augmentation import build_box_augmentor
from psd2.structures import Boxes, Instances, ImageList, pairwise_iou, BoxMode
from psd2.layers import ShapeSpec, batched_nms,FrozenBatchNorm2d
from psd2.utils.events import get_event_storage
from psd2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from psd2.modeling.proposal_generator import build_proposal_generator
from psd2.modeling.id_assign import build_id_assigner
from psd2.layers.mem_matching_losses import OIMLoss
from psd2.modeling.extend.clip_model import build_CLIP_from_openai_pretrained,Transformer,ResidualAttentionBlock
import copy
from collections import OrderedDict
import torch.utils.checkpoint as ckpt

logger = logging.getLogger(__name__)

# GeneralizedRCNN as reference
@META_ARCH_REGISTRY.register()
class ClipSimple(SearchBaseTBPS):
    @configurable
    def __init__(
        self,
        clip_model,
        freeze_at_stage2,
        proposal_generator,
        roi_heads,
        id_assigner,
        bn_neck,
        oim_loss,
        cws,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.clip_model=clip_model
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner = id_assigner
        self.bn_neck = bn_neck
        self.bn_neck_text=copy.deepcopy(bn_neck)
        self.oim_loss = oim_loss
        self.cws=cws

        # res5 only in roi_head
        for p in self.clip_model.visual.layer4.parameters():
            p.requires_grad=False
        visual_encoder=self.clip_model.visual
        if freeze_at_stage2:
            for m in [visual_encoder.conv1, visual_encoder.conv2,  visual_encoder.conv3, visual_encoder.layer1]:
                for p in m.parameters():
                    p.requires_grad=False
            visual_encoder.bn1=convert_frozen_bn(visual_encoder.bn1)
            visual_encoder.bn2=convert_frozen_bn(visual_encoder.bn2)
            visual_encoder.bn3=convert_frozen_bn(visual_encoder.bn3)
            visual_encoder.layer1=convert_frozen_bn(visual_encoder.layer1)
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE,text_dropout=cfg.PERSON_SEARCH.DET.CLIP.TEXT_DROPOUT,frozen_stem=cfg.PERSON_SEARCH.DET.CLIP.FROZEN_STEM)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":32}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        roi_feat_spatial=[roi_feat_spatial[0]//2,roi_feat_spatial[1]//2]
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4)
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = OIMLoss(cfg)
        ret["cws"]=cfg.PERSON_SEARCH.REID.CWS
        return ret
    def backbone(self,x):
        visual_encoder=self.clip_model.visual
        def stem(x):
            for conv, bn in [(visual_encoder.conv1, visual_encoder.bn1), (visual_encoder.conv2, visual_encoder.bn2), (visual_encoder.conv3, visual_encoder.bn3)]:
                x = visual_encoder.relu(bn(conv(x)))
            x = visual_encoder.avgpool(x)
            return x
        # x = ckpt.checkpoint(stem,x)
        x=stem(x)
        x =ckpt.checkpoint(visual_encoder.layer1,x) if (self.training and not visual_encoder.stem_frozen) else visual_encoder.layer1(x)
        # x =visual_encoder.layer1(x)
        x =ckpt.checkpoint(visual_encoder.layer2,x) if self.training else visual_encoder.layer2(x)
        x =ckpt.checkpoint(visual_encoder.layer3,x) if self.training else visual_encoder.layer3(x)
        return {"res4":x}
    def img_embes(self,roi_feats):
        return self.clip_model.visual.attnpool(roi_feats)
    def text_embeds(self,text):
        # text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        text_feats=self.clip_model.encode_text(text)
        text_feats=text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        text_feats=self.bn_neck_text(text_feats)
        return text_feats
    def img_emb_infer(self,roi_feats):
        return self.clip_model.visual.attnpool(roi_feats)
    def compute_sdm(self,features1, features2, pid1,pid2, logit_scale, epsilon=1e-8):
        pid1_set=set(pid1.cpu().numpy().tolist())
        pid2_set=set(pid2.cpu().numpy().tolist())
        remove_id_1=pid1_set-pid2_set
        remove_id_2=pid2_set-pid1_set
        if len(remove_id_1)>0:
            keep_mask1=torch.ones(features1.shape[0])
            for rid in list(remove_id_1):
                keep_mask1[pid1==rid]=0.
            keep_mask1=keep_mask1==1
            features1=features1[keep_mask1]
            pid1=pid1[keep_mask1]
        if len(remove_id_2)>0:
            keep_mask2=torch.ones(features2.shape[0])
            for rid in list(remove_id_2):
                keep_mask2[pid2==rid]=0.
            keep_mask2=keep_mask2==1
            features2=features2[keep_mask2]
            pid2=pid2[keep_mask2]
        """
        Similarity Distribution Matching
        """
        pid1 = pid1.reshape((features1.shape[0], 1)) # make sure pid size is [batch_size, 1]
        pid2=pid2.reshape((features2.shape[0], 1)) # make sure pid size is [batch_size, 1]
        pid_dist = pid1 - pid2.t() # n1 x n2
        labels_1_2 = (pid_dist == 0).float()
        labels_2_1 = labels_1_2.t()

        f1_norm = features1 / features1.norm(dim=1, keepdim=True)
        f2_norm = features2 / features2.norm(dim=1, keepdim=True)

        f2f1_cosine_theta = f2_norm @ f1_norm.t()
        f1f2_cosine_theta = f2f1_cosine_theta.t()

        f2_proj_f1 = logit_scale * f2f1_cosine_theta
        f1_proj_f2 = logit_scale * f1f2_cosine_theta

        # normalize the true matching distribution
        labels_1_2_distribute = labels_1_2 / labels_1_2.sum(dim=1,keepdim=True)
        labels_2_1_distribute = labels_2_1 / labels_2_1.sum(dim=1,keepdim=True)

        f1f2_pred = F.softmax(f1_proj_f2, dim=1)
        f1f2_loss = f1f2_pred * (F.log_softmax(f1_proj_f2, dim=1) - torch.log(labels_1_2_distribute + epsilon))
        f2f1_pred = F.softmax(f2_proj_f1, dim=1)
        f2f1_loss = f2f1_pred * (F.log_softmax(f2_proj_f1, dim=1) - torch.log(labels_2_1_distribute + epsilon))
        # if torch.isnan(f1f2_loss).sum()>0 or torch.isnan(f2f1_loss).sum()>0:
        #    print("get")
        loss = torch.mean(torch.sum(f1f2_loss, dim=1)) + torch.mean(torch.sum(f2f1_loss, dim=1))

        return loss
    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances
        )

        if self.training:
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            text_tokens=[]
            text_pids=[]
            for img_gt in gt_instances:
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        num_desc=len(p_tokens)
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            box_embs = self.img_emb_infer(box_features)
            box_embs = self.bn_neck(box_embs)
            box_embs=F.normalize(box_embs,dim=-1)
            # nms
            reid_feats = torch.split(
                box_embs, [len(instances_i) for instances_i in pred_instances]
            )
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, gt_i, reid_feats_i in zip(
                pred_instances, gt_instances, reid_feats
            ):
                pred_boxes_t = pred_i.pred_boxes.tensor
                pred_scores = pred_i.pred_scores
                filter_mask = pred_scores >= score_t
                pred_boxes_t = pred_boxes_t[filter_mask]
                pred_scores = pred_scores[filter_mask]
                cate_idx = pred_scores.new_zeros(
                    pred_scores.shape[0], dtype=torch.int64
                )
                # nms
                keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
                pred_boxes_t = pred_boxes_t[keep]
                pred_scores = pred_scores[keep]
                pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
                pred_i.pred_scores = pred_scores
                pred_feats=reid_feats_i[filter_mask][keep]
                if self.cws:
                    pred_feats=pred_feats*pred_scores.view(-1,1)
                pred_i.reid_feats = pred_feats
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances

    def forward_query(self, image_list, gt_instances):
        # one sentece for each query
        text_tokens=[]
        for inst in gt_instances:
            text_tokens.append(inst.descriptions[0][0])
        text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
        text_embs = F.normalize(text_embs, dim=-1)
        text_embs = torch.split(text_embs, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=text_embs[i])
            for i in range(len(text_embs))
        ]

@META_ARCH_REGISTRY.register()
class ClipViper(ClipSimple):
    @configurable
    def __init__(
        self,
        rm_min,
        rm_max,
        ech_min,
        ech_max,
        mim_ratio,
        mim_w,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.rm_min=rm_min
        self.rm_max=rm_max
        self.ech_min=ech_min
        self.ech_max=ech_max
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = FullyCrossAttentionTransformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        embed_dim=self.clip_model.text_projection.shape[1]*2
        self.mim_norm=nn.LayerNorm(embed_dim//2)
        self.mim_head=nn.Linear(embed_dim//2,embed_dim)
        self.mim_token=nn.Parameter(torch.zeros(1,1, embed_dim))
        trunc_normal_(self.mim_token, mean=0., std=.02)
        self.mim_ratio=mim_ratio
        self.mim_w=mim_w
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        box_aug = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        prev_head=ret["roi_heads"]
        res5=copy.deepcopy(prev_head.res5)
        head = ClipRes5ROIHeadsPsSpatial(cfg, res5,box_aug,res_output_shape)
        ret["roi_heads"]=head
        ret["rm_min"]=cfg.PERSON_SEARCH.REID.ATTN.RM_MIN
        ret["rm_max"]=cfg.PERSON_SEARCH.REID.ATTN.RM_MAX
        ret["ech_min"]=cfg.PERSON_SEARCH.REID.ATTN.ECH_MIN
        ret["ech_max"]=cfg.PERSON_SEARCH.REID.ATTN.ECH_MAX
        ret["mim_ratio"]=cfg.PERSON_SEARCH.REID.MIM_RATIO
        if hasattr(cfg.PERSON_SEARCH.REID,"MIM_LOSS_W"):
            mim_w=cfg.PERSON_SEARCH.REID.MIM_LOSS_W
        else:
            mim_w=1.0
        ret["mim_w"]=mim_w
        return ret
    def cross_former(self, q, k, v,with_ckpt=False):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=with_ckpt)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def _contextmix(self,roi_tokens,attn):
        attn_cost=((attn.unsqueeze(1)-attn.unsqueeze(0))**2).sum(-1)
        dig_idxs=list(range(attn.shape[0]))
        attn_cost[dig_idxs,dig_idxs]=1e10
        match_row,match_col=linear_sum_assignment(attn_cost.cpu().numpy())
        perm_tokens=roi_tokens[match_col]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_mix_tokens=torch.randint(self.ech_min,self.ech_max+1,(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        org_lembs=super().img_embes(roi_feats)
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        all_tokens_ech=self._contextmix(all_tokens,x_attn)

        def inner_forward(x,attn_mask):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            if attn_mask is not None:
                attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            else:
                attn_mask_head=None
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False,
                attn_mask=attn_mask_head,
            )
            return x

        img_feats_ech=inner_forward(all_tokens_ech,None)
        embs_ech=img_feats_ech[0]

        sort_attn_idxs=torch.argsort(x_attn,dim=-1)
        num_mask_tokens=torch.randint(self.rm_min,self.rm_max+1,(all_tokens.shape[0],))
        t_attn_value=x_attn[list(range(all_tokens.shape[0])),sort_attn_idxs[list(range(all_tokens.shape[0])),(-num_mask_tokens).cpu().numpy().tolist()].cpu().numpy().tolist()]
        attn_mask[...,1:][(x_attn>=t_attn_value.unsqueeze(1)).unsqueeze(1).expand(-1,attn_mask.shape[1],-1)]=float("-inf")
        img_feats_rm=inner_forward(all_tokens,attn_mask)
        embs_rm=img_feats_rm[0]
        
        return torch.cat([org_lembs,embs_ech,embs_rm],dim=0),torch.cat([roi_ids_uni_ulb,roi_ids_uni_ulb,roi_ids_uni_ulb],dim=0)
    def random_masked_tokens_and_labels(self,all_tokens): # before attention pooling
        prob = torch.rand(all_tokens.shape[:2],device=all_tokens.device).unsqueeze(2) # B x L x 1
        masking_w=torch.zeros_like(prob)
        masking_w[prob<self.mim_ratio]=1
        mim_tokens=self.mim_token.expand(all_tokens.shape[0],all_tokens.shape[1],-1)
        masked_tokens=all_tokens*(1-masking_w)+mim_tokens*masking_w
        def inner_forward(x):
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.clip_model.visual.attnpool.num_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=ckpt.checkpoint(inner_forward,masked_tokens.transpose(0,1))
        _,feats=img_feats[0],img_feats[1:]
        return feats.transpose(0,1),masked_tokens
    def mim_loss(self,i_features,tokens):
        masked_i_features,masked_org_features=self.random_masked_tokens_and_labels(i_features)
        text_feats =self.clip_model.encode_text_zero_dropout(torch.stack(tokens).to(self.device),ckpt=True)
        rec_i_features =self.cross_former(masked_i_features, text_feats, text_feats,with_ckpt=True)
        rec_i_features =self.mim_head(self.mim_norm(rec_i_features))
        tgt_i_features=i_features.detach()
        l2_dist=F.mse_loss(rec_i_features,tgt_i_features,reduction="mean")

        return {"loss_mim":l2_dist*self.mim_w}
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            next_pse_ulbs=1e5
            pse_gt_pids=[]
            for inst in gt_instances:
                pse_gt_pids_i=copy.deepcopy(inst.gt_pids)
                for i,pid in enumerate(inst.gt_pids):
                    if pid==-1:
                        pse_gt_pids_i[i]=next_pse_ulbs
                        next_pse_ulbs+=1
                pse_gt_pids.append(pse_gt_pids_i)

            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                pse_gt_pids, # [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,assign_ids = self.img_embes(box_features,assign_ids)
            box_feats=box_features.flatten(2,3).permute(2,0,1).contiguous()
            box_embs = self.bn_neck(box_embs)
            assign_ids[assign_ids>=1e5]=-1
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            img_features=[]
            dense_text_tokens=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_feats=img_rois[roi_ids==pid]
                        num_desc=len(p_tokens)
                        ex_id_feats=id_feats.unsqueeze(1).expand(-1,num_desc,-1,-1).flatten(0,1)
                        ex_text_tokens=list(itertools.chain(*([p_tokens]*id_feats.shape[0])))
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
                        img_features.append(ex_id_feats)
                        dense_text_tokens.extend(ex_text_tokens)
            if len(img_features)==0:
                losses["loss_mim"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mim_loss(roi_feats,dense_text_tokens))
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)

            for i, instances_i in enumerate(pred_instances):
                ids_i=assign_ids_per_img[i]
                ids_i[ids_i>=1e5]=-1
                instances_i.assign_ids = ids_i

            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
    
class ClipRes5ROIHeadsPs(Res5ROIHeads):
    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return ckpt.checkpoint(self.res5,x) if self.training else self.res5(x)
    @classmethod
    def from_config(cls, cfg,res5, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["box_predictor"] = (
            FastRCNNOutputLayersPs(
                cfg, ShapeSpec(channels=2048, height=1, width=1)
            )
        )
        ret["res5"]=res5
        return ret
    @classmethod
    def _build_res5_block(cls, cfg):
        return nn.Identity(),2048 # trivial impl
    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        num_gts=gt_classes.numel()
        has_gt = num_gts > 0
        
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels_with_gt(
            gt_classes,
            self.batch_size_per_image,
            self.positive_fraction,
            self.num_classes,
            num_gts
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        # NOTE self.proposal_append_gt is True by default
        proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        pos_match_indices = []  # for reid label assign
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            src_idxs = torch.arange(
                0,
                len(proposals_per_image),
                dtype=torch.int64,
                device=match_quality_matrix.device,
            )  # src_idxs are the indices after sampling
            tgt_idxs = matched_idxs[sampled_idxs]
            pos_mask = gt_classes < self.num_classes
            src_idxs = src_idxs[pos_mask]
            tgt_idxs = tgt_idxs[pos_mask]  # make it compatible with detr-like matchers
            pos_match_indices.append((src_idxs, tgt_idxs))

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt, pos_match_indices
    def box_embedding(self,box_features):
        return box_features.mean(dim=[2, 3])
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        
        box_embs = self.box_embedding(box_features)
        predictions = self.box_predictor(box_embs)

        if self.training:

            losses = self.box_predictor.losses(predictions, proposals)
            with torch.no_grad():
                # for vis and id_assign
                pred_instances = self.box_predictor.inference_unms(
                    predictions, proposals
                )
            del features
            return (
                pred_instances,
                box_features,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, box_features, {}, None


class FastRCNNOutputLayersPs(FastRCNNOutputLayers):
    def inference_unms(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        all_boxes = self.predict_boxes(predictions, proposals)
        all_scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        results = []
        # borrowed from fast_rcnn_inference
        for scores, boxes, image_shape in zip(all_scores, all_boxes, image_shapes):
            valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(
                dim=1
            )
            if not valid_mask.all():
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]

            scores = scores[:, :-1]
            num_bbox_reg_classes = boxes.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

            # 1. Filter results based on detection scores. It can make NMS more efficient
            #    by filtering out low-confidence detections.
            filter_mask = scores >= 0.0  # R x K NOTE disable this
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero()
            if num_bbox_reg_classes == 1:
                boxes = boxes[filter_inds[:, 0], 0]
            else:
                boxes = boxes[filter_mask]
            scores = scores[filter_mask]

            # 2. Apply NMS for each class independently. NOTE disable this

            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            result.pred_scores = scores
            result.pred_classes = filter_inds[:, 1]
            results.append(result)
        return results


class ClipRes5ROIHeadsPsSpatial(ClipRes5ROIHeadsPs):
    @configurable
    def __init__(
        self,
        box_aug,
        *args,
        **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self.box_aug=box_aug
    @classmethod
    def from_config(cls, cfg,res5,box_aug, input_shape):
        ret = super().from_config(cfg,res5,input_shape)
        ret["box_aug"]=box_aug
        return ret
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )
            """
            proposals: list of Instances with
            image_size, proposal_boxes, objectness_logits, gt_classes, gt_boxes, gt_pids
            pos_match_indices: list of
            (src_idxs, tgt_idxs)
            """
            for i,gts_i in enumerate(targets):
                gt_boxes=gts_i.gt_boxes.tensor
                match_idxs=torch.arange(gts_i.gt_pids.shape[0],device=gts_i.gt_pids.device)
                aug_boxes, aug_match_idxs = self.box_aug.augment_boxes(
                    [gt_boxes],
                    [match_idxs],
                    det_boxes=None,
                    det_pids=None,
                    img_sizes=[gts_i.image_size],
                )
                aug_boxes,aug_match_idxs=aug_boxes[0],aug_match_idxs[0]
                num_aug=aug_boxes.shape[0]
                num_prev=proposals[i].proposal_boxes.tensor.shape[0]
                proposals[i].proposal_boxes.tensor=torch.cat([proposals[i].proposal_boxes.tensor,aug_boxes])
                proposals[i].objectness_logits=torch.cat([proposals[i].objectness_logits,torch.ones(num_aug,device=aug_boxes.device)])
                proposals[i].gt_classes=torch.cat([proposals[i].gt_classes,gts_i.gt_classes[aug_match_idxs]])
                p_gt_boxes=proposals[i].gt_boxes.tensor
                aug_gt_boxes=gt_boxes[aug_match_idxs]
                proposals[i].gt_boxes.tensor=torch.cat([p_gt_boxes,aug_gt_boxes])
                proposals[i].gt_pids=torch.cat([proposals[i].gt_pids,gts_i.gt_pids[aug_match_idxs]])
                
                src_idxs=torch.cat( [pos_match_indices[i][0],torch.arange(num_aug,device=aug_boxes.device)+num_prev])
                tgt_idxs=torch.cat( [pos_match_indices[i][1],aug_match_idxs])
                pos_match_indices[i]=(src_idxs,tgt_idxs)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        
        box_embs = self.box_embedding(box_features)
        predictions = self.box_predictor(box_embs)

        if self.training:

            losses = self.box_predictor.losses(predictions, proposals)
            with torch.no_grad():
                # for vis and id_assign
                pred_instances = self.box_predictor.inference_unms(
                    predictions, proposals
                )
            del features
            return (
                pred_instances,
                box_features,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, box_features, {}, None

class FullyCrossAttentionTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualFullyCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.pos_q=None
        self.pos_k=None
    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,with_ckpt=False,return_inter=False,attn_mask=None):
        inter_out=[]
        if self.pos_q is not None:
            q=q+self.pos_q.unsqueeze(1)
        if with_ckpt:
            # def inner_forward()
            for blk in self.resblocks:
                if self.pos_k is not None:
                    k=k+self.pos_k.unsqueeze(1)
                q=ckpt.checkpoint(blk,q,k,v,attn_mask)
                if return_inter:
                    inter_out.append(q)
        else:
            for blk in self.resblocks:
                if self.pos_k is not None:
                    k=k+self.pos_k.unsqueeze(1)
                q=blk(q,k,v,attn_mask)
                if return_inter:
                    inter_out.append(q)
        if return_inter:
            q=inter_out
        return q

class ResidualFullyCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,attn_amsk=None):
        # cross attn
        q = q + self.attn(self.ln_1(q),k,v,need_weights=False, attn_mask=attn_amsk)[0]
        q = q + self.mlp(self.ln_2(q))
        return q



from psd2.layers import nonzero_tuple
def subsample_labels_with_gt(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int,num_gts:int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive= nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    gt_indices=torch.arange(0,num_gts,device=positive.device)+positive.numel()-num_gts
    perm1=torch.randperm(positive.numel()-num_gts, device=positive.device)[:num_pos-num_gts]
    perm1=torch.cat([perm1,gt_indices])
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

def convert_frozen_bn(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = FrozenBatchNorm2d(module.num_features,module.eps)
        module_output.weight=module.weight.data
        module_output.bias=module.bias.data
        module_output.running_mean=module.running_mean
        module_output.running_var=module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, convert_frozen_bn(child))
    del module
    return module_output

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

import warnings

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor