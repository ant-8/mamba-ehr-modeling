import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import os
import gzip
import json
from tqdm import tqdm

class PretrainDataset(Dataset):
    def __init__(self, directory="./dataset", max_seq_len=4096):
        self.data = []
        self.tokenizer = Tokenizer()
        for filename in tqdm(os.listdir(directory), desc="Loading/Tokenizing"):
            if filename.endswith('.json.gz'):
                filepath = os.path.join(directory, filename)
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    _data = json.load(f)
                    for subject in _data:
                        if len(subject["admissions"]) > 0:
                            encoded = self.tokenizer.encode(subject, max_seq_len)
                            if len(encoded.concept_ids) < 15:
                                continue
                            self.data.append(encoded)

        self.data = sorted(self.data, key=lambda x: len(x.concept_ids), reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """if isinstance(idx, list) and False:
            result = {
                "concept_ids": [],
                "age_ids": [],
                "time_ids": [],
                "segment_ids": [],
                "visit_order_ids": [],
            }
            for i in idx:
                subject = self.data[i]
                encoded = self.tokenizer.encode(subject)
                result["concept_ids"].append(encoded.concept_ids)
                result["age_ids"].append(encoded.age_ids)
                result["time_ids"].append(encoded.time_ids)
                result["segment_ids"].append(encoded.segment_ids)
                result["visit_order_ids"].append(encoded.visit_order_ids)
            
            for key in result.keys():
                result[key] = pad_sequences(result[key], self.tokenizer.pad_token_id)
                #result[key] = torch.tensor(result[key])
            
            return result
        else:"""
        encoded = self.data[idx]
        result = {
            "concept_ids": encoded.concept_ids,
            "age_ids": encoded.age_ids,
            "time_ids": encoded.time_ids,
            "segment_ids": encoded.segment_ids,
            "visit_order_ids": encoded.visit_order_ids
        }
        return result