from scipy.io import loadmat
from os.path import join as opj

import numpy as np
from psd2.structures import BoxMode

# from psd2.utils.logger import setup_logger
from tqdm import tqdm
import json
import logging

logger = logging.getLogger(__name__)  # setup_logger()
__all__ = ["subsets", "load_prw"]
subsets = ("Train", "Query", "Gallery", "TrainXi")
id_remap_dict = {}
for i in range(1, 479):
    id_remap_dict[i] = i - 1
for i in range(480, 484):
    id_remap_dict[i] = i - 2
for i in range(484, 932):
    id_remap_dict[i] = i
id_remap_dict[932] = 482
id_remap_dict[479] = 483
id_remap_dict[933] = 932
id_remap_dict[-2] = -1

debug_names = ["c1s2_071096.jpg"]


def load_prw_tbps(dataset_dir, subset=subsets[0]):
    """
    Returns:
        train/test set meta dict
            [
                {
                    "file_name": image paths,
                    "image_id": image name,
                    "annotations":
                        [
                            {
                                "bbox": person xyxy_abs boxes,
                                "bbox_mode": format of bbox
                                "person_id":  person id
                            },...
                        ]
                },
                ...
            ]
        or
        query meta dict
            [
                {
                    "query":
                        {
                            "file_name": image paths,
                            "image_id": image name,
                            "annotations":
                                [
                                    {
                                        "bbox": person xyxy_abs boxes,
                                        "bbox_mode": format of bbox
                                        "person_id":  person id
                                    }
                                ]
                        }
                }
            ]
    """
    assert subset in subsets

    if subset == subsets[0]:
        return _load_subset_full_anno(dataset_dir, "train")
    if subset == subsets[-1]:
        return _load_subset_full_anno_xi(dataset_dir, "train")
    elif subset == subsets[1]:
        q_dicts = []
        logger.info("Loading PRW query info :")
        with open(opj(dataset_dir, "query_info.txt"), "r") as qf:
            qinfo_lines = qf.readlines()
        with open(opj(dataset_dir,"PRW-TBPS_test"),"r") as tf:
            query_text=json.load(tf)
        imgorgid_to_text={}
        for item in query_text:
            imgorgid_to_text[item["pic_path"]+"_"+str(item["id"])]=item["desrciption"]
        with tqdm(total=len(qinfo_lines)) as pbar:
            for line in qinfo_lines:
                line = line.strip()
                if len(line) > 0:
                    infos = line.split(" ")
                    pid = int(infos[0])
                    box = np.array(infos[1:-1], dtype=np.float32)
                    box[2:] = box[:2] + box[2:]
                    img_path = opj(dataset_dir, "frames", infos[-1] + ".jpg")
                    q_dicts.extend(
                        [{
                            "query": {
                                "file_name": img_path,
                                "image_id": infos[-1] + ".jpg",
                                "annotations": [
                                    {
                                        "bbox": box,
                                        "bbox_mode": BoxMode.XYXY_ABS,
                                        "person_id": get_resort_id(pid),
                                        "descriptions": [des],
                                    }
                                ],
                            }
                        } for des in imgorgid_to_text[infos[-1] + ".jpg_"+str(pid)]]
                    )
                pbar.update(1)
        return q_dicts
    else:
        return _load_subset_full_anno(dataset_dir, "test")


def get_resort_id(labeled_id):
    global id_remap_dict
    """global id_remap_dict
    global id_next
    if labeled_id < 0:
        return -1
    elif labeled_id in id_remap_dict.keys():

        return id_remap_dict[labeled_id]
    else:
        resort_id = id_next
        id_remap_dict[labeled_id] = resort_id
        id_next += 1
        return resort_id"""
    return id_remap_dict[labeled_id]


def _load_subset_full_anno(dataset_dir, subset_name):

    sub_f_mat = loadmat(opj(dataset_dir, "frame_" + subset_name + ".mat"))[
        "img_index_" + subset_name
    ].squeeze()
    sub_img_fns = sub_f_mat.tolist()
    subset_dicts = []
    logger.info("Loading PRW {} :".format(subset_name))
    if subset_name==subsets[0].lower():
        with open(opj(dataset_dir,"PRW-TBPS_train"),"r") as tf:
            train_text=json.load(tf)
        imgorgid_to_text={}
        for item in train_text:
            imgorgid_to_text[item["pic_path"]+"_"+str(item["id"])]=item["desrciption"]

    with tqdm(total=len(sub_img_fns)) as pbar:
        for img_fn in sub_img_fns:
            img_fn = img_fn[0]
            img_name = img_fn + ".jpg"
            img_anno_mat = loadmat(opj(dataset_dir, "annotations", img_name + ".mat"))
            ks = sorted(img_anno_mat.keys())
            img_anno = img_anno_mat[ks[-1]]
            boxes = img_anno[:, 1:].copy().astype(np.float32)  # n x 4 xywh-boxes
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            ids = img_anno[:, 0].astype(np.int32).copy()
            if subset_name==subsets[0].lower():
                texts=[imgorgid_to_text[img_name+"_"+str(orgid)] if orgid>0 else ["A person with unknown identity."] for orgid in ids] # more like description padding for unlabeled person
                subset_dicts.append(
                    {
                        "file_name": opj(dataset_dir, "frames", img_name),
                        "image_id": img_name,
                        "annotations": [
                            {
                                "bbox": boxes[i],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "person_id": get_resort_id(ids[i]),
                                "descriptions": texts[i]
                            }
                            for i in range(boxes.shape[0])
                        ],
                    }
                )
            else:
                subset_dicts.append(
                    {
                        "file_name": opj(dataset_dir, "frames", img_name),
                        "image_id": img_name,
                        "annotations": [
                            {
                                "bbox": boxes[i],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "person_id": get_resort_id(ids[i]),
                            }
                            for i in range(boxes.shape[0])
                        ],
                    }
                )
            pbar.update(1)
    return subset_dicts

def _load_subset_full_anno_xi(dataset_dir, subset_name):

    sub_f_mat = loadmat(opj(dataset_dir, "frame_" + subset_name + ".mat"))[
        "img_index_" + subset_name
    ].squeeze()
    sub_img_fns = sub_f_mat.tolist()
    subset_dicts = []
    logger.info("Loading PRW {} :".format(subset_name))
    if subset_name==subsets[0].lower():
        with open(opj(dataset_dir,"PRW-TBPS_train"),"r") as tf:
            train_text=json.load(tf)
        imgorgid_to_text={}
        imgorgid_to_img={}
        for item in train_text:
            imgorgid_to_text[item["pic_path"]+"_"+str(item["id"])]=item["desrciption"]
            if item["id"] in imgorgid_to_img:
                imgorgid_to_img[item["id"]].append(item["pic_path"])
            else:
                imgorgid_to_img[item["id"]]=[item["pic_path"]]

    with tqdm(total=len(sub_img_fns)) as pbar:
        for img_fn in sub_img_fns:
            img_fn = img_fn[0]
            img_name = img_fn + ".jpg"
            img_anno_mat = loadmat(opj(dataset_dir, "annotations", img_name + ".mat"))
            ks = sorted(img_anno_mat.keys())
            img_anno = img_anno_mat[ks[-1]]
            boxes = img_anno[:, 1:].copy().astype(np.float32)  # n x 4 xywh-boxes
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            ids = img_anno[:, 0].astype(np.int32).copy()
            if subset_name==subsets[0].lower():
                texts=[imgorgid_to_text[img_name+"_"+str(orgid)] if orgid>0 else ["A person with unknown identity."] for orgid in ids] # more like description padding for unlabeled person
                xi_texts=[]
                for i in ids:
                    if i>0:
                        id_pics=imgorgid_to_img[i]
                        temp_texts=[]
                        for pic in id_pics:
                            if pic != img_name:
                                pic_id_texts=imgorgid_to_text[pic+"_"+str(i)]
                                temp_texts.extend(pic_id_texts)
                        xi_texts.append(temp_texts)
                    else:
                        xi_texts.append(["A person with unknown identity."])
                subset_dicts.append(
                    {
                        "file_name": opj(dataset_dir, "frames", img_name),
                        "image_id": img_name,
                        "annotations": [
                            {
                                "bbox": boxes[i],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "person_id": get_resort_id(ids[i]),
                                "descriptions": texts[i],
                                "xi_descriptions": xi_texts[i]
                            }
                            for i in range(boxes.shape[0])
                        ],
                    }
                )
            else:
                subset_dicts.append(
                    {
                        "file_name": opj(dataset_dir, "frames", img_name),
                        "image_id": img_name,
                        "annotations": [
                            {
                                "bbox": boxes[i],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "person_id": get_resort_id(ids[i]),
                            }
                            for i in range(boxes.shape[0])
                        ],
                    }
                )
            pbar.update(1)
    return subset_dicts