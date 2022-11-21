# coding: utf-8
import multiprocessing
import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import check_and_make_path, get_format_str, get_nx_graph


class StructureInfoGenerator:
    base_path: str
    origin_base_path: str
    core_base_path: str
    node_num: int

    def __init__(self, base_path, origin_folder, core_folder, node_file):
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.core_base_path = os.path.abspath(os.path.join(base_path, core_folder))

        check_and_make_path(self.core_base_path)
        
    def get_nx_graph(self, file_path, sep='\t'):
        df = pd.read_csv(file_path, sep=sep)
        if df.shape[1] == 2:
            df['weight'] = 1.0
        graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='weight', create_using=nx.Graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph
    # Most real-world graphs' k-core numbers start from 0. However, SBM graphs' k-core numbers start from 40(or 70), not start from 0.
    # Moreover, most real world graphs' k-core numbers are 0, 1, 2, 3, ... without any interval, while synthetic graph such as SBM's k-core numbers are 46,48,51,54, ..
    def get_kcore_graph(self, input_file, output_dir, sep='\t', core_list=None, degree_list=None):
        input_path = os.path.join(self.origin_base_path, input_file)
        graph = self.get_nx_graph(input_path, sep=sep)
        core_num_dict = nx.core_number(graph)
        print("unique core nums: ", len(np.unique(np.array(list(core_num_dict.values())))))
        max_core_num = max(list(core_num_dict.values()))
        print('file name: ', input_file, 'max core num: ', max_core_num)

        check_and_make_path(output_dir)
        node_num=max(graph.nodes)+1
        format_str = get_format_str(max_core_num)
        print('out', output_dir)
        for i in range(1, max_core_num + 1):
            node_list=[i for i in range(node_num)]
            k_core_graph = nx.k_core(graph, k=i, core_number=core_num_dict)
            k_core_graph.add_nodes_from(node_list)
            A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=node_list)
            signature = format_str.format(i)
            sp.save_npz(os.path.join(output_dir, signature + '.npz'), A)

    def get_kcore_graph_all_time(self, sep='\t', worker=-1):
        print("getting k-core sub-graphs for all timestamps...")

        f_list = os.listdir(self.origin_base_path)
        f_list = sorted(f_list)

        core_list, degree_list = [], []
        for i, f_name in enumerate(f_list):
            self.get_kcore_graph(input_file=f_name, output_dir=os.path.join(self.core_base_path, f_name.split('.')[0]), sep=sep,
                                    core_list=core_list, degree_list=degree_list)
