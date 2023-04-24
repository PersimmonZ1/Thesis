# Name: cenet_model
# Author: Reacubeth
# Time: 2021/6/25 17:28
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
import math
import copy


# 二分类器的模型，预测一个样本的实体是否在其历史实体集中
class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim),  # 批标准化
                                    nn.Dropout(0.4),  # 随机将一部分输入张量设为0
                                    nn.LeakyReLU(0.2),  # 对每个元素使用LeakyReLU计算
                                    nn.Linear(input_dim, out_dim),
                                    )

    def forward(self, x):
        return self.linear(x)


class CENET(nn.Module):
    # 完成对网络各层的定义和初始化，包括权重和嵌入的初始化
    def __init__(self, num_e, num_rel, num_t, args):
        super(CENET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args

        # entity relation embedding
        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, args.embedding_dim))  # 将参数添加到Module参数列表中
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))  # 均匀分布初始化，gain=根号2
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, args.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        # 做线性变换，将n*num_e的输入矩阵变换成n*dim的输出矩阵
        # 是查询表征的一部分
        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)

        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.oracle_layer = Oracle(3 * args.embedding_dim, 1)  # 二分类器
        self.oracle_layer.apply(self.weights_init)  # 初始化Oracle模型线性层的权重参数

        self.linear_pred_layer_s1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        # 初始化线性层的权重参数
        self.weights_init(self.linear_frequency)
        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)

        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()  # 对Softmax值添加对数操作
        self.softmax = nn.Softmax()  # 输入输出张量维度相同
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()  # 二元交叉熵损失函数
        self.oracle_mode = args.oracle_mode  # 硬掩码或软掩码

        print('CENET Initiated')

    @staticmethod
    # 用均值初始化线性变换子模块的权重参数
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_block, mode_lk, total_data=None):
        quadruples, s_history_event_o, o_history_event_s, \
            s_history_label_true, o_history_label_true, s_frequency, o_frequency = batch_block

        # 没有历史信息返回损失为None，如果样本都为训练集中第一个时间戳的样本就没有历史信息，即从训练集第二个时间戳开始训练
        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, obj_rank, batch_loss = [None] * 3
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        # 利用频率向量生成Z向量，同时计算查询表征的一部分
        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax
        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_non_history_tag[s_history_tag == 1] = -self.args.lambdax
        o_non_history_tag[o_history_tag == 1] = -self.args.lambdax
        s_non_history_tag[s_history_tag == 0] = self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_frequency = F.softmax(s_frequency, dim=1)  # 对每一行的频率向量使用softmax函数
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))  # 查询表征的一部分
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        # 训练模式返回两个损失的加权
        if mode_lk == 'Training':
            # 计算历史依赖和非历史依赖的损失函数，论文中是求和，实际上两个损失都是求平均
            s_nce_loss, _ = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],  # 宾语实体预测
                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                    s_history_tag, s_non_history_tag)
            o_nce_loss, _ = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                    o_history_tag, o_non_history_tag)
            # calculate_spc_loss(self, hidden_lk, actor1, r, rel_embeds, targets):
            # 计算监督对比损失
            s_spc_loss = self.calculate_spc_loss(s, r, self.rel_embeds[:self.num_rel],
                                                 s_history_label_true, s_frequency_hidden)
            o_spc_loss = self.calculate_spc_loss(o, r, self.rel_embeds[self.num_rel:],
                                                 o_history_label_true, o_frequency_hidden)
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            return self.args.alpha * nce_loss + (1 - self.args.alpha) * spc_loss

        # 返回三种情况下计算的损失之和、宾语预测和主语预测的排名，以及二分类器分类正确的样本数：
        # 三种情况：使用二分类器生成的掩码向量、不使用掩码向量、使用基本事实生成的掩码向量
        elif mode_lk in ['Valid', 'Test']:
            # 记录每个样本的历史实体集
            s_history_oid = []
            o_history_sid = []

            for i in range(quadruples.shape[0]):
                s_history_oid.append([])
                o_history_sid.append([])
                for con_events in s_history_event_o[i]:
                    s_history_oid[-1] += con_events[:, 1].tolist()
                for con_events in o_history_event_s[i]:
                    o_history_sid[-1] += con_events[:, 1].tolist()

            # 得到历史非历史依赖预测的损失和概率
            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                          self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                          s_history_tag, s_non_history_tag)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                          self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                          o_history_tag, o_non_history_tag)

            # 得到二分类器预测的损失和概率，以及该组中预测正确的样本数量
            s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)

            # 根据二分类器分类结果和样本的历史实体集生成掩码向量
            s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            for i in range(quadruples.shape[0]):
                if s_pred_history_label[i].item() > 0.5:
                    s_mask[i, s_history_oid[i]] = 1
                else:
                    s_mask[i, :] = 1
                    s_mask[i, s_history_oid[i]] = 0

                if o_pred_history_label[i].item() > 0.5:
                    o_mask[i, o_history_sid[i]] = 1
                else:
                    o_mask[i, :] = 1
                    o_mask[i, o_history_sid[i]] = 0

            if self.oracle_mode == 'soft':
                s_mask = F.softmax(s_mask, dim=1)
                o_mask = F.softmax(o_mask, dim=1)


            s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask, total_data, 's', True)
            o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask, total_data, 'o', True)
            batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0

            # 不使用掩码向量计算损失和排名
            s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask, total_data, 's', False)
            o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask, total_data, 'o', False)
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0

            # Ground Truth，根据基本事实和历史实体集生成掩码向量
            s_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            for i in range(quadruples.shape[0]):
                if o[i] in s_history_oid[i]:
                    s_mask_gt[i, s_history_oid[i]] = 1
                else:
                    s_mask_gt[i, :] = 1
                    s_mask_gt[i, s_history_oid[i]] = 0

                if s[i] in o_history_sid[i]:
                    o_mask_gt[i, o_history_sid[i]] = 1
                else:
                    o_mask_gt[i, :] = 1
                    o_mask_gt[i, o_history_sid[i]] = 0

            s_total_loss3, sub_rank3 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask_gt, total_data, 's', True)
            o_total_loss3, obj_rank3 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask_gt, total_data, 'o', True)
            batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0

            return sub_rank1, obj_rank1, batch_loss1, \
                   sub_rank2, obj_rank2, batch_loss2, \
                   sub_rank3, obj_rank3, batch_loss3, \
                   (s_ce_all_acc + o_ce_all_acc) / 2

        elif mode_lk == 'Oracle':
            print('Oracle Training')
            s_ce_loss, _, _ = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                               s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                               o_history_label_true, o_frequency_hidden)
            return (s_ce_loss + o_ce_loss) / 2.0 + self.oracle_l1(0.01)  # 返回的损失为交叉熵损失+范数

    # 训练二分类器的损失函数
    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):
        history_label_pred = F.sigmoid(
            self.oracle_layer(torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()  # 断开Tensor与计算图的连接，并且不再需要梯度
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        print('# CE Accuracy', ce_accuracy)
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        return ce_loss, history_label_pred, ce_accuracy * tmp_label.shape[0]

    # 计算历史依赖和非历史依赖的损失
    def calculate_nce_loss(self, actor1, actor2, r, rel_embeds, linear1, linear2, history_tag, non_history_tag):
        preds_raw1 = self.tanh(linear1(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))  # 线性层包括权重矩阵和偏置矩阵
        preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1)) + history_tag, dim=1)  # 计算H向量并执行Softmax操作

        preds_raw2 = self.tanh(linear2(  # 历史依赖和非历史依赖使用两个线性层
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds2 = F.softmax(preds_raw2.mm(self.entity_embeds.transpose(0, 1)) + non_history_tag, dim=1)

        # 选取每一行中目标宾语的概率值，相加取对数，再求和得到总损失，除以样本数
        nce = torch.sum(torch.gather(torch.log(preds1 + preds2), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]

        pred_actor2 = torch.argmax(preds1 + preds2, dim=1)  # predicted result，得到一维张量
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print('# Batch accuracy', accuracy)  # 输出该组样本的预测正确率

        return nce, preds1 + preds2

    # 计算历史非历史依赖损失+二元交叉熵损失的总损失；计算每个实体在非时间感知过滤设置下的排名
    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, trust_musk, all_triples, pred_known, oracle,
                     history_tag=None, case_study=False):
        if case_study:
            f = open("case_study.txt", "a+")
            entity2id, relation2id = get_entity_relation_set(self.args.dataset)

        if oracle:
            preds = torch.mul(preds, trust_musk)
            print('$Batch After Oracle accuracy:', end=' ')
        else:
            print('$Batch No Oracle accuracy:', end=' ')
        # compute the correct triples
        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result，返回最大值索引一维张量
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print(accuracy)

        total_loss = nce_loss + ce_loss

        # 计算每个实体在非时间感知过滤设置下的排名
        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]
            if case_study:
                in_history = torch.where(history_tag[i] > 0)[0]
                not_in_history = torch.where(history_tag[i] < 0)[0]
                print('---------------------------', file=f)
                for hh in range(in_history.shape[0]):
                    print('his:', entity2id[in_history[hh].item()], file=f)

                print(pred_known,
                      'Truth:', entity2id[cur_s.item()], '--', relation2id[cur_r.item()], '--', entity2id[cur_o.item()],
                      'Prediction:', entity2id[pred_actor2[i].item()], file=f)

            o_label = cur_o
            ground = preds[i, cur_o].clone().item()  # 正确答案的预测概率
            # 非时间感知过滤设置
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]

                preds[i, idx] = 0
                preds[i, o_label] = ground

            # 得到两个布尔数组，若概率相等取中值排名
            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        return total_loss, ranks

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.rel_embeds.pow(2)) + torch.mean(self.entity_embeds.pow(2))
        return regularization_loss * reg_param

    # 所有参数绝对值之和*系数作为范数
    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.oracle_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    # contrastive
    def freeze_parameter(self):
        self.rel_embeds.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)
        self.contrastive_output_layer.requires_grad_(False)

    # MLP解码器，并没有进行标准化且将嵌入映射到单位球上
    def contrastive_layer(self, x):
        # Implement from the encoder E to the projection network P
        # x = F.normalize(x, dim=1)
        x = self.contrastive_hidden_layer(x)
        # x = F.relu(x)
        # x = self.contrastive_output_layer(x)
        # Normalize to unit hypersphere
        # x = F.normalize(x, dim=1)
        return x

    # 计算监督对比损失
    def calculate_spc_loss(self, actor1, r, rel_embeds, targets, frequency_hidden):
        projections = self.contrastive_layer(  # 查询表征
            torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)  # 本来是列向量，去除为1的维度，变成一维张量，也就是行向量
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0  # 温度参数设为1.0等于没有
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        # 这部分处理原论文未提及
        exp_dot_tempered = (
                # torch.max返回最大值和索引，0是取最大值
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        # 得到N*N的矩阵，再判断相等，得到判断样本缺失实体和其他实体类型是否相同的掩码
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))  # 创建一个对角线为0，其他位置为1的N*N矩阵
        mask_combined = mask_similar_class * mask_anchor_out  # 排除样本自身
        cardinality_per_samples = torch.sum(mask_combined, dim=1)  # 每个样本相同类型实体的基数
        # N*N张量，每个点积除以该除的分母再取log，原论文忘记给分母加指数了
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        # N*1张量，得到每个样本的监督对比损失
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss
