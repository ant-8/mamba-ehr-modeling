import json
import numpy as np
import torch
from datetime import datetime

BIN_ID_OFFSET = 5

class Metadata:
    def __init__(self, age=0, time=-1, segment=0, visit_order=-1):
        self.age = age
        self.time = time
        self.segment = segment
        self.visit_order = visit_order

    def __repr__(self):
        return f"METADATA: AGE={self.age}, TIME={self.time}, SEGMENT={self.segment}, VISIT_ORDER={self.visit_order}"

class Token:
    def __init__(self, id, metadata=None):
        self.id = id
        if metadata is None:
            self.metadata = Metadata()
        else:
            self.metadata = metadata

    def __repr__(self):
        return f"{self.id} "
    
    def __str__(self):
        return f"[\n\tID={self.id}\n\t{repr(self.metadata)}\n]"

class TokenSequence:
    def __init__(self, metadata):
        self.ids = []
        self.metadata = metadata

    def add_token_id(self, id):
        self.ids.append(id)

    def __len__(self):
        return len(self.tokens)
    
    def get_tokens(self):
        return [
            Token(id, self.metadata)
            for id in self.ids
        ]

    def join(seq_list):
        result = []

        for seq in seq_list:
            result.extend(seq.get_tokens())
        
        return result
    
class EncodingOutput:
    def __init__(
        self, tokens, concept_ids, age_ids,
        time_ids, segment_ids, visit_order_ids
    ):
        self.tokens = tokens
        self.concept_ids = concept_ids
        self.age_ids = age_ids
        self.time_ids = time_ids
        self.segment_ids = segment_ids
        self.visit_order_ids = visit_order_ids


class Tokenizer:
    def __init__(self, config_dir="./tokenizer.json"):
        with open(config_dir, 'rt', encoding='utf-8') as f:
            config = json.load(f)
            self.bins = config["bins"]
            self.vocab = config["vocab"]
            self.vocab_size = config["vocab_size"]

        self.pad_token_id = self.vocab["PAD"]
        self.cls_token_id = self.vocab["CLS"]
        self.bos_token_id = self.vocab["BOS"]

    def get_age_bin_id(self, age):
        bin_edges = np.array(self.bins["age"])
        bin_indice = np.digitize(age, bin_edges)
        assert bin_indice >= 0
        bin_indice = min(bin_indice, 4)
        return bin_indice + BIN_ID_OFFSET
    
    def get_lab_test_value_bin_id(self, lab_test_id, value):
        bin_edges = self.bins["lab_tests"].get(str(lab_test_id), None)
        if value is None or bin_edges is None: return None
        try:
            value = float(value)
        except ValueError:
            return None
        bin_indice = np.digitize(value, bin_edges) - 1
        """try:
            assert bin_indice >= 0
        except Exception:
            print(bin_edges, lab_test_id, value)
            raise Exception"""
        bin_indice = max(bin_indice, 0)
        bin_indice = min(bin_indice, 4)
        return bin_indice + BIN_ID_OFFSET

    def encode_admission(self, admission, metadata):
        seq = TokenSequence(metadata)

        seq.add_token_id(self.vocab["ADM_START"])
        seq.add_token_id(self.vocab["AGE"])

        #elta_year = int(admission["admittime"][:4]) - subject['anchor_year']
        #age += delta_year
        #seq.add_token_id(self.get_age_bin_id(age))

        seq.add_token_id(self.vocab["LAB_TEST"])
        for test in admission["lab_tests"]:
            bin_id = self.get_lab_test_value_bin_id(test["itemid"], test["value"])
            if bin_id is not None:
                seq.add_token_id(self.vocab["LAB_TEST_IDS"][str(test["itemid"])])
                seq.add_token_id(bin_id)

        seq.add_token_id(self.vocab["PROCEDURE"])
        for procedure in admission["procedures"]:
            seq.add_token_id(self.vocab["PROCEDURE_ICD_CODES"][procedure["icd_code"]])

        seq.add_token_id(self.vocab["ADM_END"])
        return seq.get_tokens()

    def encode(self, subject, max_seq_len=4096):
        result = []
        result.append(Token(id=self.vocab["BOS"]))

        parsed_admissions = [(adm, datetime.strptime(adm['admittime'], '%Y-%m-%d %H:%M:%S')) for adm in subject['admissions']]
        parsed_admissions.sort(key=lambda x: x[1])

        subject["admissions"] = [adm[0] for adm in parsed_admissions]
        adm_times = [adm[1] for adm in parsed_admissions]
        
        if len(adm_times) > 0:
            delta_times = [0] + [(adm_times[i+1] - adm_times[i]).days for i in range(len(adm_times) - 1)]
        else:
            delta_times = []
            
        visit_order = 1

        assert len(delta_times) == len(subject["admissions"])
        for i in range(len(subject["admissions"])):
            admission = subject["admissions"][i]
            delta_year = int(admission["admittime"][:4]) - subject['anchor_year']
            age = subject["anchor_age"] + delta_year
            segment = 2 if visit_order % 2 == 0 else 1
            md = Metadata(age=age, visit_order=visit_order, time=delta_times[i], segment=segment)

            result.extend(self.encode_admission(admission, md))
            visit_order += 1
        result = result[:max_seq_len-1]
        result.append(Token(id=self.vocab["CLS"]))        
        return EncodingOutput(
            tokens=result,
            concept_ids=[ x.id for x in result ],
            age_ids=[ x.metadata.age for x in result ],
            time_ids=[ x.metadata.time for x in result ],
            segment_ids=[ x.metadata.segment for x in result ],
            visit_order_ids=[ x.metadata.visit_order for x in result ]
        )