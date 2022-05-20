# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
import os
import sys
import typing

import detectron2.engine.launch
import numpy as np
import torch
from src.object_detect_fr.common_util.custom_evaluator import ObjDetectEvaluator
from src.templates_common.base_conf import LOGGER
from src.templates_common.cv_util import constants
from src.templates_common.cv_util.dataset_util import register_dataset_to_dt2
from src.templates_common.cv_util.dt2_util import setup_cfg, get_trainer
from src.templates_common.cv_util.model import Dataset
from src.templates_common.cv_util.model_train import parse_and_save_train_result, parse_eval_metrics_lis


def train_by_dt2(dataset_test, dataset_train, train_args_inst):
    num_gpus_per_machine = torch.cuda.device_count()
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    detectron2.engine.launch(main, num_gpus_per_machine, 1, machine_rank=0, dist_url="tcp://127.0.0.1:{}".format(port),
                             args=(dataset_train, dataset_test, num_gpus_per_machine, train_args_inst.num_epochs,
                                   train_args_inst.score_thresh_test, train_args_inst.nms_thresh_test),
                             )


def main(dataset_train: Dataset, dataset_test: Dataset, num_gpus: int, num_epochs: int, score_thresh_test: float,
         nms_thresh_test: float):
    # 配置安装
    model_yaml_path = os.path.join(os.path.join(constants.MODEL_DIR, 'configs'),
                                   'configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    pkl_path = os.path.join(constants.MODEL_DIR, 'model_final_f6e8b1.pkl')
    cfg = setup_cfg(num_gpus, num_epochs, score_thresh_test, nms_thresh_test,
                    train_size=len(dataset_train.images), category_num=len(dataset_train.category_set.category_dic),
                    model_yaml_path=model_yaml_path, pkl_path=pkl_path)

    # 注册数据集 到 detectron2
    register_dataset_to_dt2(dataset_train, dataset_test)

    # 获取训练器
    eval_metrics_dic: typing.Dict[str, float] = {}
    trainer = get_trainer(cfg, ObjDetectEvaluator, eval_metrics_dic)
    trainer.resume_or_load(resume=True)
    trainer.train()
    eval_metrics_lis: typing.List[str] = trainer.eval_metrics_lis
    LOGGER.info('------------ train end ------------')
    parse_and_save_train_result(cfg, eval_metrics_dic, eval_metrics_lis, parse_eval_metrics)


def parse_eval_metrics(eval_metrics_dic_in: typing.Dict[str, float],
                       eval_metrics_list_in: typing.List[str]):
    """
    :param eval_metrics_dic_in:
    样例 {'AP': 62.56810681068108,
         'AP50': 100.0,
         'AP75': 79.04290429042904,
         'APs': nan,
         'APm': 62.485148514851474,
         'APl': 68.31683168316832,
         'AP-feet': 59.1989198919892,
         'AP-hat': 65.93729372937294
         }
    :param eval_metrics_list_in: like
    样例 ['Average Precision(AP)IoU0.50:0.95all100at0.6256810681068108',
         'Average Precision(AP)IoU0.50all100at1.0',
         'Average Precision(AP)IoU0.75all100at0.7904290429042904',
         'Average Precision(AP)IoU0.50:0.95small100at-1',
         'Average Precision(AP)IoU0.50:0.95medium100at0.6248514851485147',
         'Average Precision(AP)IoU0.50:0.95large100at0.6831683168316832',
         'Average Recall(AR)IoU0.50:0.95all1at0.5',
         'Average Recall(AR)IoU0.50:0.95all10at0.6583333333333333',
         'Average Recall(AR)IoU0.50:0.95all100at0.6583333333333333',
         'Average Recall(AR)IoU0.50:0.95small100at-1',
         'Average Recall(AR)IoU0.50:0.95medium100at0.6712121212121213',
         'Average Recall(AR)IoU0.50:0.95large100at0.6833333333333333']
    :return:
    """
    LOGGER.info(f"get_eval_metrics\n{eval_metrics_dic_in=}\n{eval_metrics_list_in=}")
    label_metrics_lis = []
    for ap_name, ap_value in eval_metrics_dic_in.items():
        if ap_name[:3] == "AP-":
            label_name = ap_name[3:]
            if np.isnan(ap_value):
                ap_value = ''
            label_metrics_lis.append({'label': label_name, 'boxAp': ap_value})

    bbox_dic, seg_dic = parse_eval_metrics_lis(eval_metrics_list_in)
    return label_metrics_lis, bbox_dic, seg_dic
