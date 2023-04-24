# Name: valid
# Author: Reacubeth
# Time: 2021/8/25 10:30
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import argparse
import numpy as np
import torch
import pickle
import time
import datetime
import os
import random
import utils
from cenet_model import CENET


# 返回不使用掩码向量和使用基本事实相关掩码向量两种设置下，得到的验证集所有样本的排名
# 当前还未训练二分类器，返回第一个设置下样本排名无意义
def execute_valid(args, total_data, model,
                  data,
                  s_history, o_history,
                  s_label, o_label,
                  s_frequency, o_frequency):
    # 不使用掩码向量，得到的所有验证集样本的排名
    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    # 使用基本事实生成的掩码向量，得到的所有验证集样本的排名
    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []  # 所有样本的排名
    total_data = utils.to_device(torch.from_numpy(total_data))
    for batch_data in utils.make_batch(data,
                                       s_history,
                                       o_history,
                                       s_label,
                                       o_label,
                                       s_frequency,
                                       o_frequency,
                                       args.batch_size):
        batch_data[0] = utils.to_device(torch.from_numpy(batch_data[0]))
        for i in [3, 4, 5, 6]:
            batch_data[i] = utils.to_device(torch.from_numpy(batch_data[i])).float()

        with torch.no_grad():
            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Valid', total_data)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3
