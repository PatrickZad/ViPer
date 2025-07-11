import psd2.data.transforms as dT
import torchvision.transforms as tT

import numpy as np

from ..detection_utils import read_image


import numbers
import warnings

import math
from psd2.structures import Instances, Boxes, BoxMode
import torch
from psd2.utils.simple_tokenizer import SimpleTokenizer


class SearchMapper(object):
    def __init__(self, cfg, is_train) -> None:
        self.is_train = is_train
        self.in_fmt = cfg.INPUT.FORMAT
        if self.is_train:
            self.augs = dT.AugmentationList(
                [
                    dT.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        size_divisibility=1, # cfg.INPUT.SIZE_DIVISIBILITY, ImageList takes care
                        sample_style="choice",
                    ),
                    dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                ]
            )
        else:
            self.augs = dT.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                size_divisibility=1, # cfg.INPUT.SIZE_DIVISIBILITY, ImageList takes care
                sample_style="choice",
            )
        self.totensor = (
            tT.ToTensor()
        )  # tT.Normalize(mean=img_mean, std=img_std) is moved to model: preprocess_input
        if cfg.INPUT.REA.ENABLED:
            self.rea = RandomInstanceErasingTensor()
        else:
            self.rea = trivial_rea

    def __call__(self, img_dict):
        """
        Args:
            {
                "file_name": full path,
                "image_id": image name,
                 "annotations": [
                        K x
                        {
                            "bbox": single box,
                            "bbox_mode": BoxMode,
                            "person_id": person id,
                        }
                    ],
            }
            or
            {
                "query:
                    {
                        "file_name": full path,
                        "image_id": image name,
                        "annotations":
                            [
                                K x {
                                    "bbox": single box,
                                    "bbox_mode": BoxMode,
                                    "person_id": person id,
                                }
                            ],
                    }
                "gallery": (optional)
                    [
                        N x {
                            "file_name": full path,
                            "image_id": image name,
                            "annotations":
                                [
                                    1 x {
                                        "bbox": single box,
                                        "bbox_mode": BoxMode,
                                        "person_id": person id,
                                    }
                                    or
                                    1 x {
                                        "bbox": empty [],
                                        "bbox_mode": BoxMode,
                                        "person_id": None,
                                    }
                                ],
                        }
                    ]

            }
        Outs:
            {
                "image":image tensor in (0,1), unnormed,
                "height": h,
                "width": w,
                "file_name": image ful path,
                "image_id": image name,
                "gt_boxes": augmented Boxes,
                "gt_pids": person ids tensor,
                "gt_classes": class ids tensor,
                "org_img_size": original image size (oh,ow),
                "org_gt_boxes": original Boxes,
            }
            or
            {
                "query":
                    {
                        "image":image tensor in (0,1), unnormed,
                        "height": h,
                        "width": w,
                        "file_name": image ful path,
                        "image_id": image name,
                        "gt_boxes": augmented Boxes,
                        "gt_pids": person ids tensor,
                        "gt_classes": class ids tensor,
                        "org_img_size": original image size (oh,ow),
                        "org_gt_boxes": original Boxes,
                    }
                "gallery":
                    [
                        N x {
                            "file_name": image ful path,
                            "image_id": image name,
                            "gt_pids": person ids tensor,
                            "org_gt_boxes": original Boxes,
                        }
                    ]
            }
        """
        if "query" in img_dict.keys():
            rst = {}
            rst["query"] = self._transform_annotations(img_dict["query"])
            if "gallery" in img_dict:
                rst["gallery"] = []
                for gdict in img_dict["gallery"]:
                    gann = gdict["annotations"][0]
                    box_mode = gann["bbox_mode"]
                    boxes = gann["bbox"]
                    img_path = gdict["file_name"]
                    if box_mode != BoxMode.XYXY_ABS and boxes.size > 0:
                        img_arr = read_image(img_path, self.in_fmt)
                        boxes = Boxes(boxes, box_mode).convert_mode(
                            BoxMode.XYXY_ABS, img_arr.shape[:2]
                        )
                    else:
                        # boxes=torch.zeros((0,4))
                        boxes = Boxes(boxes, BoxMode.XYXY_ABS)
                    g_instance = Instances(
                        None,
                        file_name=img_path,
                        image_id=gdict["image_id"],
                        org_gt_boxes=boxes,
                        gt_pids=torch.tensor([gann["person_id"]], dtype=torch.int)
                        if gann["person_id"] is not None
                        else None,
                    )
                    rst["gallery"].append(g_instance)
            return rst
        else:
            return self._transform_annotations(img_dict)

    def _transform_annotations(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.in_fmt)
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box_mode = ann["bbox_mode"]
            boxes.append(
                Boxes(ann["bbox"], box_mode)
                .convert_mode(BoxMode.XYXY_ABS, img_arr.shape[:2])
                .tensor[0]
                .tolist()
            )
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs(aug_input)
        aug_img = aug_input.image
        h, w = aug_img.shape[:2]
        aug_boxes = aug_input.boxes
        img_t = self.totensor(aug_img.copy())
        self.rea(img_t, aug_boxes)
        return {
            "image": img_t,
            "instances": Instances(
                (h, w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
            ),
        }


class RandomInstanceErasingTensor:
    """Randomly selects a rectangle region in a person and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against bounding box.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.

    Returns:
        Erased Image.

    """

    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=[0.485, 0.456, 0.406],
    ):
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence"
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = torch.as_tensor(scale)
        self.ratio = torch.as_tensor(ratio)
        self.value = torch.as_tensor(value)

    def get_params(self, box, channels, scale, ratio, value=None):
        """Get parameters for ``erase`` for a random erasing."""
        box = torch.round(box).astype(torch.int32)
        img_c, img_h, img_w = channels, box[3] - box[1], box[2] - box[0]
        area = img_h * img_w

        log_ratio = torch.log(ratio)
        for _ in range(10):

            erase_area = (
                area * (torch.rand(1) * (scale[1] - scale[0]) + scale[0]).item()
            )
            aspect_ratio = torch.exp(
                (torch.rand(1) * (log_ratio[1] - log_ratio[0]) + log_ratio[0])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.normal(0.0, 1.0, size=(img_c, h, w))
            else:
                v = value.view(img_c, 1, 1)

            i = torch.randint(0, img_w - w + 1, size=1).item()
            j = torch.randint(0, img_h - h + 1, size=1).item()
            return i + box[0].item(), j + box[1].item(), h, w, v

        # Return original image
        return box[0].item(), box[1].item(), 0, 0, 0

    def __call__(self, img, boxes):
        vals = torch.rand(boxes.shape[0])
        for i in range(boxes.shape[0]):
            if vals[i] < self.p:
                x, y, h, w, v = self.get_params(
                    boxes[i], 3, scale=self.scale, ratio=self.ratio, value=self.value
                )
                self.erase(img, x, y, h, w, v)

    def erase(self, img, i, j, h, w, v):

        img[j : j + h, i : i + w] = v


def trivial_rea(img, ins_boxes):
    pass

class SearchMapperTextBased(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg,is_train)
        self.tokenizer=SimpleTokenizer()
        self.max_num_text=cfg.INPUT.MAX_QUERY_NUM
    def tokenize(self,caption: str, text_length=77, truncate=True) -> torch.LongTensor:
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(caption) + [eot_token]

        result = torch.zeros(text_length, dtype=torch.long)
        if len(tokens) > text_length:
            if truncate:
                tokens = tokens[:text_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {caption} is too long for context length {text_length}"
                )
        result[:len(tokens)] = torch.tensor(tokens)
        return result
    def _transform_annotations(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.in_fmt)
        boxes = []
        ids = []
        texts=[]
        texts_xi=[]
        for ann in img_dict["annotations"]:
            box_mode = ann["bbox_mode"]
            boxes.append(
                Boxes(ann["bbox"], box_mode)
                .convert_mode(BoxMode.XYXY_ABS, img_arr.shape[:2])
                .tensor[0]
                .tolist()
            )
            ids.append(ann["person_id"])
            if "descriptions" in ann:
                valid_ts=[]
                for t in ann["descriptions"]:
                    if len(t)>1:
                        valid_ts.append(self.tokenize(t))
                texts.append(valid_ts)
                if "xi_descriptions" in ann and len(valid_ts)<self.max_num_text:
                    valid_xi_ts=[]
                    for t in ann["xi_descriptions"]:
                        if len(t)>1:
                            valid_xi_ts.append(self.tokenize(t))
                    if len(valid_ts)+len(valid_xi_ts)>self.max_num_text:
                        select_idxs=torch.randperm(len(valid_xi_ts))[:self.max_num_text-len(valid_ts)].numpy().tolist()
                        valid_xi_ts=[valid_xi_ts[i] for i in select_idxs]
                    texts_xi.append(valid_xi_ts)
            
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs(aug_input)
        aug_img = aug_input.image
        h, w = aug_img.shape[:2]
        aug_boxes = aug_input.boxes
        img_t = self.totensor(aug_img.copy())
        self.rea(img_t, aug_boxes)
        return {
            "image": img_t,
            "instances": Instances(
                (h, w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
                descriptions=texts,
                xi_descriptions=texts_xi,
            ),
        }

class SearchMapperMmq(SearchMapperTextBased):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg,is_train)
        if self.is_train:
            self.augs_q = dT.AugmentationList(
                [
                    dT.Resize( cfg.INPUT.QUERY_SIZE),
                    dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                ]
            )
        else:
            self.augs_q = dT.Resize( cfg.INPUT.QUERY_SIZE)
        self.num_query=cfg.INPUT.MAX_QUERY_NUM
    def _transform_annotations(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.in_fmt)
        boxes = []
        ids = []
        texts=[]
        qimgs=[]
        for ann in img_dict["annotations"]:
            if ann["person_id"]==-1:
                continue # TODO use unlabeled
            box_mode = ann["bbox_mode"]
            boxes.append(
                Boxes(ann["bbox"], box_mode)
                .convert_mode(BoxMode.XYXY_ABS, img_arr.shape[:2])
                .tensor[0]
                .tolist()
            )
            ids.append(ann["person_id"])
            if "descriptions" in ann:
                valid_ts=[]
                for ts in ann["descriptions"]:
                    text_tokens=[self.tokenize(ti) for ti in ts if len(ti)>1]
                    valid_ts.append(text_tokens)
                if len(valid_ts)>self.num_query:
                    select_idxs=torch.randperm(len(valid_ts))[:self.num_query].numpy().tolist()
                else:
                    select_idxs=list(range(len(valid_ts)))
                valid_ts=[valid_ts[i] for i in select_idxs]
                texts.append(valid_ts)
            if "queries" in ann:
                selected=[ann["queries"][i] for i in select_idxs]
                qimgs_i=[]
                for qn in selected:
                    qimg=read_image(qn, self.in_fmt)
                    aug_input = dT.AugInput(image=qimg.copy())
                    transforms = self.augs_q(aug_input)
                    aug_qimg = aug_input.image
                    qimgs_i.append(self.totensor(aug_qimg.copy()))
                qimgs.append(qimgs_i)
            
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs(aug_input)
        aug_img = aug_input.image
        h, w = aug_img.shape[:2]
        aug_boxes = aug_input.boxes
        img_t = self.totensor(aug_img.copy())
        self.rea(img_t, aug_boxes)
        #TODO multiple text
        return {
            "image": img_t,
            "instances": Instances(
                (h, w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
                descriptions=texts,
                queries=qimgs
            ),
        }