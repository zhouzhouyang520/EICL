
import math
import tensorflow as tf
import csv
import numpy as np
import tqdm
import pickle

import datetime
import time
import faiss
from collections import Counter

#from EmpatheticIntents.model import *
#from EmpatheticIntents.utilities import *
#from EmpatheticIntents.optimize import CustomSchedule
#from EmpatheticIntents.create_datasets import create_datasets

import os
from data_processor.utils.tools.time_warpper import wrapper_calc_time

tf.compat.v1.enable_eager_execution()

class Retriver():

    @wrapper_calc_time()
    def __init__(self, topk, train_vectors, test_vectors):
        self.topk = topk
        print(f"The topk of Ann: {self.topk}")

        # Load data and emotion
#        print(f"Converting train tensor to numpy...")
#        train_vectors = train_vectors.detach().cpu().numpy()
#        print(f"Converting test tensor to numpy...")
#        test_vectors = test_vectors.detach().cpu().numpy()
        self.train_vectors = np.array(train_vectors)
        self.test_vectors = np.array(test_vectors)
        print(f"Building Ann index...")
        self.index = self.build_ann_index(self.train_vectors)
        print(f"Building Ann index end...")

    # The following methods are deprecated and no longer used
    # def load_np(self, cache_file):
    #     data = np.load(cache_file, allow_pickle=True)
    #     return data
    # 
    # def load_np_data(self, train_name, dev_name, test_name):
    #     print("Loading emotion list...")
    #     data_dir = config.data_dir
    #     train = self.load_np(data_dir + "/" + train_name)
    #     dev = self.load_np(data_dir + "/" + dev_name)
    #     test = self.load_np(data_dir + "/" + test_name)
    #     return train, dev, test
    # 
    # def load_pickle_data(self, data_name):
    #     print("Loading siuation vector ...")
    #     data_dir = config.data_dir
    #     cache_file = f"{data_dir}/{data_name}"
    #     if os.path.exists(cache_file):
    #         print(f"Loading data: {cache_file } ...")
    #         with open(cache_file, "rb") as f:
    #             [data_tra, data_val, data_tst] = pickle.load(f)
    #     return data_tra, data_val, data_tst

    @wrapper_calc_time(print_log=True)
    def build_ann_index(self, datas):
        print("Building Ann index ...")
        print("datas shape:", datas.shape)
        dim = datas.shape[1]
        ids = np.array(list(range(datas.shape[0]))).astype(np.int64) #[i for i in range(datas.shape[0])])
        print("ids:", ids)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(quantizer)
    
        faiss.normalize_L2(datas)
        index.add_with_ids(datas,ids)
        #print("index.is_trained:", index.is_trained)
        print("Index.ntotal:", index.ntotal)
        return index

    def search_one(self, index, vec):
        vec = np.expand_dims(vec, axis=0) 
        reuslt_weight, results = index.search(vec, self.topk)
        #print("results:", len(results), results, reuslt_weight)
        return results, reuslt_weight
    
    @wrapper_calc_time(print_log=True)
    def search(self, test_vec):
        index_result = []
        #test_vec = data["sit_vec"][:10] # For test
        for i, vec in enumerate(test_vec):
            results, reuslt_weight = self.search_one(self.index, vec)
            index_result.append((results, reuslt_weight))
        #print(f"test_vec: {len(test_vec)}, index_result: {index_result}")
        return index_result

#if __name__ == "main":
#    retriver = Retriver(topk=50, train_vectors, test_vectors)
#    print("Test dev data...")
#    retriver.search(retriver.sit_val)
#    print("Test test data...")
#    retriver.search(retriver.sit_tst)
