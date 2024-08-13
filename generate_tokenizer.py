import json
import gzip
import os
import itertools
import numpy as np
from tqdm import tqdm

directory = './dataset/'
dataset = []

for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.json.gz'):
        filepath = os.path.join(directory, filename)
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            dataset.extend(data)

all_admissions = list(itertools.chain(*[ x['admissions'] for x in dataset ]))
all_ages = [ int(x['anchor_age']) for x in dataset ]

all_lab_tests = list(itertools.chain(*[ x['lab_tests'] for x in all_admissions ]))
all_lab_test_ids = list(set([ x['itemid'] for x in all_lab_tests ]))

all_procedures = list(itertools.chain(*[ x['procedures'] for x in all_admissions ]))
all_procedure_icd_codes = list(set([ x['icd_code'] for x in all_procedures ]))

def is_numeric(value):
    if value is None: return False
    if '-' in value and value.index('-') > 0: return False
    return value.replace('.', '', 1).isdigit() or value.replace('.', '', 1).replace('-', '', 1).isdigit()

def create_bin(values, n_bins=5):
    if len(values) < n_bins: return None
    bin_edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    return np.round(bin_edges, 2).tolist()

def create_bin_edges_from_test_id(lab_test_id):
    values = [ x['value'] for x in all_lab_tests if x['itemid'] == lab_test_id and is_numeric(x['value']) ]
    values = [ float(x) for x in values ]
    if len(values) == 0: return None
    return create_bin(values)

age_bin = create_bin(all_ages)

lab_test_bins = {}

from tqdm import tqdm
for id in tqdm(all_lab_test_ids):
    edges = create_bin_edges_from_test_id(id)
    if edges is None: continue
    lab_test_bins[id] = edges

bins = {
    "age": age_bin,
    "lab_tests": lab_test_bins
}

vocab = {
    "BOS": 0,
    "CLS": 1,
    "ADM_START": 2,
    "ADM_END": 3,
    "AGE": 4,
    "BIN_0": 5,
    "BIN_1": 6,
    "BIN_2": 7,
    "BIN_3": 8,
    "BIN_4": 9,
    "LAB_TEST": 10,
    "PAD": 11,
    "PROCEDURE": 12,
    "LAB_TEST_IDS": {},
    "PROCEDURE_ICD_CODES": {}
}

offset = 13
for id in all_lab_test_ids:
    vocab["LAB_TEST_IDS"][str(id)] = offset
    offset += 1

for icd in all_procedure_icd_codes:
    vocab["PROCEDURE_ICD_CODES"][icd] = offset
    offset += 1

config = {
    "vocab_size": offset,
    "vocab": vocab,
    "bins": bins
}

with open('tokenizer.json', 'w') as f:
    json.dump(config, f)