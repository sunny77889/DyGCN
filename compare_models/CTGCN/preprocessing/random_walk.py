import json
import os
import time

import numpy as np
import scipy.sparse as sp


def random_walk(spadj, walk_dir_path, freq_dir_path, f_name, walk_length, walk_time, weighted):
    # t1 = time.time()
    node_num = spadj.shape[0]
    walk_len = walk_length + 1
    spadj = spadj.tolil()
    node_neighbor_arr = spadj.rows
    node_weight_arr = spadj.data
    walk_spadj = sp.lil_matrix((node_num, node_num))
    node_freq_arr = np.zeros(node_num, dtype=int)

    weight_arr_dict = dict()
    # random walk
    #对每个节点进行随机游走
    for nidx in range(node_num):
        for iter in range(walk_time):
            walk = [nidx]
            cnt = 1
            #随机游走，路径长度为walk_len
            while cnt < walk_len:
                cur = walk[-1]
                neighbor_list = node_neighbor_arr[cur]
                if len(neighbor_list) == 0:
                    break
                if cur not in weight_arr_dict:
                    weight_arr = np.array(node_weight_arr[cur])
                    weight_arr = weight_arr / weight_arr.sum()
                    weight_arr_dict[cur] = weight_arr
                else:
                    weight_arr = weight_arr_dict[cur]
                nxt_id = np.random.choice(neighbor_list, p=weight_arr) if weighted else np.random.choice(neighbor_list)
                walk.append(int(nxt_id))
                cnt += 1
            # count walk pair
            seq_len = len(walk)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if walk[i] == walk[j]:
                        continue
                    from_id, to_id = walk[i], walk[j]
                    walk_spadj[from_id, to_id] = 1
                    walk_spadj[to_id, from_id] = 1
                    node_freq_arr[from_id] += 1
                    node_freq_arr[to_id] += 1

    tot_freq = node_freq_arr.sum()
    Z = 0.001
    neg_node_list = []
    for nidx in range(node_num):
        rep_num = int(((node_freq_arr[nidx] / tot_freq)**0.75) / Z)
        neg_node_list += [nidx] * rep_num
    walk_file_path = os.path.join(freq_dir_path, f_name.split('.')[0] + '.json')
    print('out', walk_file_path)
    with open(walk_file_path, 'w') as fp:
        json.dump(neg_node_list, fp)
    del neg_node_list, node_freq_arr

    walk_file_path = os.path.join(walk_dir_path, f_name.split('.')[0] + '.npz')
    print('out', walk_file_path)
    sp.save_npz(walk_file_path, walk_spadj.tocoo())
