import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch
import tqdm
from scipy.sparse import csc_matrix


"""
为训练集中每个时间戳上的三元组集生成图并保存到文件
记录训练验证测试集中每个样本的历史信息，历史实体频率，实体是否在历史实体集中，并保存文件
"""


# 返回数据集四元组数组，和排序后的时间戳数组
def load_quadruples(in_path, *file_names):
    quadruple_list, times = [], set()
    for file_name in file_names:
        with open(os.path.join(in_path, file_name), 'r') as fr:
            for line in fr:
                # 兼容python2要使用line.strip().split()
                head, rel, tail, time = map(int, line.split()[:4])
                quadruple_list.append([head, rel, tail, time])
                times.add(time)
    times = sorted(times)
    return np.asarray(quadruple_list), np.asarray(times)


# 返回实体关系数量
def get_total_number(in_path, file_name):
    with open(os.path.join(in_path, file_name), 'r') as fr:
        line_split = next(fr).split()[:2]
        return tuple(map(int, line_split))


def get_data_with_t(data, tim):
    triples = [quad[:3] for quad in data if quad[3] == tim]
    return np.array(triples)


# 返回所有节点入度的范数
def comp_deg_norm(g):
    return 1.0 / (g.in_degrees(range(g.number_of_nodes())).float().clamp_min(1.0))  # 设置入度的最小值为1，返回一个一维张量


# 为特定时间戳上的三元组集生成一张图：生成反向三元组，添加节点和边以及他们的属性
def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    # 将两个数组一起排序后，返回排序后的无重复元素的一维数组
    # return_inverse为True返回两个数组的所有元素在连接后的原数组的索引，因为排序后实体id从0开始，所以索引正好对应实体id
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    g = dgl.graph((src, dst))  # 初始化图，自动计算节点信息
    g.ndata['id'] = torch.from_numpy(uniq_v).long().view(-1, 1)  # 保留了原数组的信息
    g.ndata['norm'] = comp_deg_norm(g).view(-1, 1)
    g.edata['type_s'] = torch.LongTensor(np.concatenate((rel, rel + num_rels)))
    g.edata['type_o'] = torch.LongTensor(np.concatenate((rel + num_rels, rel)))
    g.ids = {id: idx for idx, id in enumerate(uniq_v)}
    return g


def get_history_target(quadruples, s_history_event_o, o_history_event_s):
    # s_history_related记录每个事件的历史相关实体的频率
    s_history_related = np.zeros((quadruples.shape[0], num_e), dtype=np.float)
    o_history_related = np.zeros((quadruples.shape[0], num_e), dtype=np.float)
    s_history_label_true = np.zeros((quadruples.shape[0], 1))
    o_history_label_true = np.zeros((quadruples.shape[0], 1))
    s_cnt, o_cnt = 0, 0  # 记录新出现的主语和新出现的宾语的数量

    for i, (s, r, o, _) in tqdm.tqdm(enumerate(quadruples), total=quadruples.shape[0]):
        s_history_oid, o_history_sid = [], []
        for con_events in s_history_event_o[i]:
            idx = (con_events[:, 0] == r).nonzero()[0]  # 得到相同关系的索引
            cur_events = con_events[idx, 1].tolist()  # 得到宾语
            s_history_oid += con_events[:, 1].tolist()  # 列表可以直接相加，此处加的是所有宾语实体，而非相同关系的宾语实体
            s_history_related[i][cur_events] += 1
        for con_events in o_history_event_s[i]:
            idx = (con_events[:, 0] == r).nonzero()[0]
            cur_events = con_events[idx, 1].tolist()
            o_history_sid += con_events[:, 1].tolist()
            o_history_related[i][cur_events] += 1

        if o in s_history_oid:
            s_history_label_true[i] = 1
        else:
            s_cnt += 1
        if s in o_history_sid:
            o_history_label_true[i] = 1
        else:
            o_cnt += 1

    # 如果宾语在主语的历史实体集中，则主语一定在宾语的历史实体集中，反之亦然。因此s_cnt和o_cnt两个变量的值相同
    print(f's {s_cnt} o {o_cnt} s_rate {s_cnt / quadruples.shape[0]:.2%} o_rate {o_cnt / quadruples.shape[0]:.2%}')
    # csc_matrix处理大型稀疏矩阵，节约存储空间
    return s_history_label_true, o_history_label_true, csc_matrix(s_history_related), csc_matrix(o_history_related)


# 得到训练集中每个时间戳上的三元组集，并生成一张图
def store_graph_dict_train(train_data, train_times, num_r):
    graph_dict_train = {}
    for tim in train_times:
        print(str(tim)+'\t'+str(max(train_times)))
        data = get_data_with_t(train_data, tim)
        graph_dict_train[tim] = get_big_graph(data, num_r)

    with open('train_graphs.txt', 'wb') as fp:
        pickle.dump(graph_dict_train, fp)

    return


# 记录数据集中每个样本头尾实体的历史相关信息，计算每个样本历史实体的频率，记录每个样本的实体是否在历史实体集中出现过。将三部分信息保存到文件中
def get_history_info(data, type, ent_his_dict, latest_t, num_e):
    s_history_data = [[] for _ in range(len(data))]  # 创建训练集四元组数量个空列表
    o_history_data = [[] for _ in range(len(data))]
    s_history_data_t = [[] for _ in range(len(data))]
    o_history_data_t = [[] for _ in range(len(data))]

    for i, sample in enumerate(data):
        if i % 10000 == 0:
            print(f'{type}', i, len(data))

        t = sample[3]
        if latest_t != t:
            for ee in range(num_e):
                if len(ent_his_dict['s_his_cache'][ee]):
                    ent_his_dict['s_his'][ee].append(ent_his_dict['s_his_cache'][ee].copy())
                    ent_his_dict['s_his_t'][ee].append(ent_his_dict['s_his_cache_t'][ee])
                    ent_his_dict['s_his_cache'][ee], ent_his_dict['s_his_cache_t'][ee] = [], None
                if len(ent_his_dict['o_his_cache'][ee]):
                    ent_his_dict['o_his'][ee].append(ent_his_dict['o_his_cache'][ee].copy())
                    ent_his_dict['o_his_t'][ee].append(ent_his_dict['o_his_cache_t'][ee])
                    ent_his_dict['o_his_cache'][ee], ent_his_dict['o_his_cache_t'][ee] = [], None
            latest_t = t
        s, r, o = sample[:3]
        s_history_data[i] = ent_his_dict['s_his'][s].copy()
        o_history_data[i] = ent_his_dict['o_his'][o].copy()
        s_history_data_t[i] = ent_his_dict['s_his_t'][s].copy()
        o_history_data_t[i] = ent_his_dict['o_his_t'][o].copy()

        # 测试集只使用训练集和验证集中的历史信息
        if type != 'test':
            if not len(ent_his_dict['s_his_cache'][s]):
                ent_his_dict['s_his_cache'][s] = np.array([[r, o]])
            else:
                ent_his_dict['s_his_cache'][s] = np.concatenate((ent_his_dict['s_his_cache'][s], [[r, o]]), axis=0)
            ent_his_dict['s_his_cache_t'][s] = t

            if not len(ent_his_dict['o_his_cache'][o]):
                ent_his_dict['o_his_cache'][o] = np.array([[r, s]])
            else:
                ent_his_dict['o_his_cache'][o] = np.concatenate((ent_his_dict['o_his_cache'][o], [[r, s]]), axis=0)
            ent_his_dict['o_his_cache_t'][o] = t

    s_label, o_label, s_history_related, o_history_related = \
        get_history_target(data, s_history_data, o_history_data)
    with open(f'{type}_history_sub.txt', 'wb') as fp:
        pickle.dump([s_history_data, s_history_data_t], fp)
    with open(f'{type}_history_ob.txt', 'wb') as fp:
        pickle.dump([o_history_data, o_history_data_t], fp)
    with open(f'{type}_s_label.txt', 'wb') as fp:
        pickle.dump(s_label, fp)
    with open(f'{type}_o_label.txt', 'wb') as fp:
        pickle.dump(o_label, fp)
    with open(f'{type}_s_frequency.txt', 'wb') as fp:
        pickle.dump(s_history_related, fp)
    with open(f'{type}_o_frequency.txt', 'wb') as fp:
        pickle.dump(o_history_related, fp)

    return latest_t


if __name__ == '__main__':
    train_data, train_times = load_quadruples('', 'train.txt')
    dev_data, dev_times = load_quadruples('', 'valid.txt')
    test_data, test_times = load_quadruples('', 'test.txt')
    num_e, num_r = get_total_number('', 'stat.txt')

    store_graph_dict_train(train_data, train_times, num_r)  # 为训练集中每个时间戳上的三元组生成一张图，存储到文件中

    ent_his_dict = {}
    ent_his_dict['s_his'] = [[] for _ in range(num_e)]  # 创建实体数量个空列表
    ent_his_dict['o_his'] = [[] for _ in range(num_e)]
    ent_his_dict['s_his_t'] = [[] for _ in range(num_e)]
    ent_his_dict['o_his_t'] = [[] for _ in range(num_e)]
    ent_his_dict['s_his_cache'] = [[] for _ in range(num_e)]
    ent_his_dict['o_his_cache'] = [[] for _ in range(num_e)]
    ent_his_dict['s_his_cache_t'] = [None for _ in range(num_e)]
    ent_his_dict['o_his_cache_t'] = [None for _ in range(num_e)]
    latest_t = 0

    latest_t = get_history_info(train_data, 'train', ent_his_dict, latest_t, num_e)
    latest_t = get_history_info(dev_data, 'dev', ent_his_dict, latest_t, num_e)
    latest_t = get_history_info(test_data, 'test', ent_his_dict, latest_t, num_e)
