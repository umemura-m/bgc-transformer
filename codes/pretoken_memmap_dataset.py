#!/usr/bin/env python3

import json,os,torch
import numpy as np
from torch.utils.data import Dataset

class PretokMemmapDataset(Dataset):
    def __init__(self,meta_json:str):
        if not os.path.isfile(meta_json):
            raise FileNotFoundError(meta_json)
        with open(meta_json,"r",encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        self.n = int(meta["num_examples"])
        self.seq_len = int(meta["seq_len"])
        self.mmap_path = meta["mmap_path"]
        # read-only memmap view
        self.arr = np.memmap(self.mmap_path,dtype=np.int32,mode="r", shape=(self.n,self.seq_len))

    def __len__(self):
        return self.n

    def __getitem__(self,i):
        # Return only input_ids; MLM collator will create labels/masks
        ids = torch.from_numpy(self.arr[i].astype(np.int64,copy=False))  # HF expects int64
        return ids

#---
