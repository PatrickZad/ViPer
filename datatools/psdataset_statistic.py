import sys

import numpy as np

sys.path.append("./")
import fire
from psd2.config import get_cfg
from psd2.data.build import get_detection_dataset_dicts

from psd2.utils.logger import setup_logger

import os


import pandas
from os.path import join as opj
import seaborn
import logging


def setup_cfg(config_file):
    setup_logger(name="psd2")
    setup_logger(name="data_test")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.SOLVER.IMS_PER_BATCH = 2

    return cfg


def stat_vis(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "statistics")
    dataset_name = cfg.DATASETS.TRAIN[0]
    train_dicts = get_detection_dataset_dicts([dataset_name])
    vis_stat(train_dicts, opj(base_out, dataset_name))
    """for dataset_name in cfg.DATASETS.TEST:
        if "InfQ" in dataset_name or "Query" in dataset_name:
            continue
        test_dicts = get_detection_dataset_dicts(dataset_name)
        vis_stat(test_dicts, opj(base_out, dataset_name))"""


def vis_stat(data_dicts, save_dir):
    log_path = opj(save_dir, "log")
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        filename=log_path,
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d%I:%M:%S %p",
    )
    n_person = []
    n_lb = []
    n_ulb = []
    id_feq = {}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if "query" in data_dicts[0].keys():
        data_dicts = [ddict["query"] for ddict in data_dicts]
    for ddict in data_dicts:
        ids = np.array(
            [ann["person_id"] for ann in ddict["annotations"]], dtype=np.int32
        )
        n_person.append(ids.shape[0])
        lb_arr = ids[ids > -1]
        n_lb.append(lb_arr.shape[0])
        n_ulb.append(ids[ids == -1].shape[0])
        for i in range(lb_arr.shape[0]):
            pid = lb_arr[i]
            if pid in id_feq:
                id_feq[pid] = id_feq[pid] + 1
            else:
                id_feq[pid] = 1
    n_freq = list(id_feq.values())
    all_vis_data = {
        "num_person": n_person,
        "num_labeled": n_lb,
        "num_unlabeled": n_ulb,
        "num_freq": n_freq,
    }
    for k, v in all_vis_data.items():
        vis_data = pandas.DataFrame({k: v})
        plt = seaborn.histplot(data=vis_data, x=k, binwidth=1)
        fig = plt.get_figure()
        fig.savefig(opj(save_dir, k + ".png"), dpi=400)
        fig.clf()
    for k, v in all_vis_data.items():
        logging.info("Average {} : {:.2f}".format(k, np.average(v)))
    for k, v in all_vis_data.items():
        logging.info("Median {} : {:.2f}".format(k, np.median(v)))


if __name__ == "__main__":
    fire.Fire()
