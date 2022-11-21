# coding: utf-8
import multiprocessing
import os
import time

import pandas as pd
from utils import check_and_make_path, get_sp_adj_mat

import preprocessing.random_walk as rw


# Random Walk Generator
class WalkGenerator:
    base_path: str
    origin_base_path: str
    walk_pair_base_path: str
    node_freq_base_path: str
    walk_time: int
    walk_length: int

    def __init__(self, base_path, origin_folder, walk_pair_folder, node_freq_folder,  node_file, walk_time=100, walk_length=5):
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        self.node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))

        self.walk_time = walk_time
        self.walk_length = walk_length

        check_and_make_path(self.walk_pair_base_path)
        check_and_make_path(self.node_freq_base_path)

    def get_walk_info(self, f_name, original_graph_path, sep='\t', weighted=True):
        print('f_name = ', f_name)
        adj = get_sp_adj_mat(original_graph_path, sep=sep)
        rw.random_walk(adj, self.walk_pair_base_path, self.node_freq_base_path, f_name, self.walk_length, self.walk_time, weighted)

    def get_walk_info_all_time(self, worker=-1, sep='\t', weighted=True):
        print("perform random walk for all file(s)...")
        f_list = os.listdir(self.origin_base_path)
        f_list = sorted(f_list)

        for i, f_name in enumerate(f_list):
            original_graph_path = os.path.join(self.origin_base_path, f_name)
            self.get_walk_info(f_name, original_graph_path=original_graph_path, sep=sep, weighted=weighted)
