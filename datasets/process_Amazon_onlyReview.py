import os
import time
import pickle
import pandas as pd
import json
import numpy as np

# # === Parameters ===
# datasets_name = 'Amazon_grocery_2018/Grocery_and_Gourmet_Food'
# time_interval = 60 * 60 * 24

# def parse(path):
#     with open(path, 'rb') as g:
#         for l in g:
#             yield json.loads(l)

# def getDF(path):
#     data = {}
#     for i, d in enumerate(parse(path)):
#         data[i] = d
#     return pd.DataFrame.from_dict(data, orient='index')


# # === Load and filter interaction data ===
# data_path = 'datasets/' + datasets_name + '.json'
# df = getDF(data_path)
# interaction = df[['reviewerID', 'asin', 'unixReviewTime']].dropna()

# # === Initial filtering: sessions with >1 interaction ===
# session_counts = interaction['reviewerID'].value_counts()
# interaction = interaction[interaction['reviewerID'].isin(session_counts[session_counts > 1].index)]

# # === Filter by 24-hour window ===
# interaction['session_start'] = interaction.groupby('reviewerID')['unixReviewTime'].transform('min')
# interaction['time_interval'] = interaction['unixReviewTime'] - interaction['session_start']
# interaction = interaction[interaction['time_interval'] < time_interval]

# # === Filter items FIRST (must have >= 10 interactions globally) ===
# item_counts = interaction['asin'].value_counts()
# items_to_keep = item_counts[item_counts >= 10].index
# interaction = interaction[interaction['asin'].isin(items_to_keep)]

# # === After item filtering, filter sessions again (sessions must have > 1 unique interaction) ===
# session_counts = interaction.groupby('reviewerID')['asin'].nunique()
# sessions_to_keep = session_counts[session_counts > 1].index
# interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]

# # === Remap sessionID and itemID from 1 ===
# interaction = interaction.rename(columns={'reviewerID': 'sessionID', 'asin': 'itemID', 'unixReviewTime': 'time'})
# session2id = {sid: idx + 1 for idx, sid in enumerate(interaction['sessionID'].unique())}
# item2id = {iid: idx + 1 for idx, iid in enumerate(interaction['itemID'].unique())}

# interaction['sessionID'] = interaction['sessionID'].map(session2id)
# interaction['itemID'] = interaction['itemID'].map(item2id)

# # === Group sequences chronologically ===
# interaction = interaction.sort_values(['sessionID', 'time'])
# sessions = interaction.groupby('sessionID')['itemID'].apply(list).to_dict()

# # === Chronological Train/test split (90/10) ===
# timestamps = interaction.groupby('sessionID')['time'].max()
# sorted_sessions = sorted(timestamps.items(), key=lambda x: x[1])
# sorted_ids = [x[0] for x in sorted_sessions]
# cutoff = int(0.9 * len(sorted_ids))

# tra_sess = {sid: sessions[sid][:20] for sid in sorted_ids[:cutoff] if len(sessions[sid]) >= 2}
# tes_sess = {sid: sessions[sid][:20] for sid in sorted_ids[cutoff:] if len(sessions[sid]) >= 2}

# # === Verify and Re-index items again for GCEGNN compatibility ===
# item_dict = {}
# item_ctr = 1

# def reindex_session(sess_dict):
#     global item_ctr
#     new_seqs = []
#     for s in sess_dict:
#         outseq = []
#         for i in sess_dict[s]:
#             if i not in item_dict:
#                 item_dict[i] = item_ctr
#                 item_ctr += 1
#             outseq.append(item_dict[i])
#         if len(outseq) >= 2:  # IMPORTANT: GCEGNN needs >=2 items per session
#             new_seqs.append(outseq)
#     return new_seqs

# tra_seqs = reindex_session(tra_sess)
# tes_seqs = reindex_session(tes_sess)

# # Generate (sequence, label) pairs
# def process_seqs_no(iseqs):
#     out_seqs, labs = [], []
#     for seq in iseqs:
#         if len(seq) >= 2:
#             out_seqs.append(seq[:-1])
#             labs.append(seq[-1])
#     return out_seqs, labs

# tr_seqs, tr_labs = process_seqs_no(tra_seqs)
# te_seqs, te_labs = process_seqs_no(tes_seqs)

# print('Sample train sequences:', tr_seqs[:5])
# print('Sample train labels:', tr_labs[:5])

# # === Final dataset ===
# train_data = (tr_seqs, tr_labs)
# test_data = (te_seqs, te_labs)

# # === Stats ===
# all_interactions = sum(len(seq) for seq in tr_seqs + te_seqs)
# print('#interactions:', all_interactions)
# print('#sessions:', len(tr_seqs) + len(te_seqs))
# print('#items:', len(item_dict))
# print('sequence average length:', all_interactions / (len(tr_seqs) + len(te_seqs)))

# # === Save ===
# output_path = "datasets/Amazon_grocery_2018/"
# os.makedirs(output_path, exist_ok=True)
# pickle.dump(train_data, open(os.path.join(output_path, "train.txt"), 'wb'))
# pickle.dump(test_data, open(os.path.join(output_path, "test.txt"), 'wb'))

# print("Dataset preprocessing done:", datasets_name)


# === Parameters ===
datasets_name = 'Amazon_grocery_2018/Grocery_and_Gourmet_Food'
time_interval = 60 * 60 * 24

def parse(path):
    with open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def getDF(path):
    data = {}
    for i, d in enumerate(parse(path)):
        data[i] = d
    return pd.DataFrame.from_dict(data, orient='index')

# === Load and filter interaction data ===
data_path = 'datasets/' + datasets_name + '.json'
df = getDF(data_path)
interaction = df[['reviewerID', 'asin', 'unixReviewTime']].dropna()

# === Filter sessions with >1 interaction ===
session_counts = interaction['reviewerID'].value_counts()
sessions_to_keep = session_counts[session_counts > 1].index
interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]

# === Filter by 24-hour window ===
session_min_time = interaction.groupby('reviewerID')['unixReviewTime'].transform('min')
interaction['time_interval'] = interaction['unixReviewTime'] - session_min_time
interaction = interaction[interaction['time_interval'] < time_interval]

# === Re-filter sessions with >1 interaction again ===
session_counts = interaction['reviewerID'].value_counts()
sessions_to_keep = session_counts[session_counts > 1].index
interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]

# === Remap sessionID and itemID ===
interaction = interaction.rename(columns={'reviewerID': 'sessionID', 'asin': 'itemID', 'unixReviewTime': 'time'})

session2id = {sid: idx + 1 for idx, sid in enumerate(interaction['sessionID'].unique())}
item2id = {iid: idx + 1 for idx, iid in enumerate(interaction['itemID'].unique())}

interaction['sessionID'] = interaction['sessionID'].map(session2id)
interaction['itemID'] = interaction['itemID'].map(item2id)

# === Filter items with at least 10 interactions ===
item_counts = interaction['itemID'].value_counts()
items_to_keep = item_counts[item_counts >= 10].index
interaction = interaction[interaction['itemID'].isin(items_to_keep)]

# === Group session sequences ===
interaction = interaction.sort_values(['sessionID', 'time'])
sessions = interaction.groupby('sessionID')['itemID'].apply(list).to_dict()

# === Train/test chronological split (90/10) ===
timestamps = interaction.groupby('sessionID')['time'].max()
sorted_sessions = sorted(timestamps.items(), key=lambda x: x[1])
sorted_ids = [x[0] for x in sorted_sessions]
cutoff = int(0.9 * len(sorted_ids))

tra_sess, tes_sess = {}, {}
for sid in sorted_ids[:cutoff]:
    if len(sessions[sid]) >= 2:
        tra_sess[sid] = sessions[sid][:20] 
for sid in sorted_ids[cutoff:]:
    if len(sessions[sid]) >= 2:
        tes_sess[sid] = sessions[sid][:20]

# === Re-index items from 1 again ===
item_dict = {}
item_ctr = 1

def reindex_session(sess_dict, train_mode=True):
    global item_ctr
    new_seqs = []
    for s in sess_dict:
        outseq = []
        for i in sess_dict[s]:
            if i in item_dict:
                outseq.append(item_dict[i])
            elif train_mode:
                item_dict[i] = item_ctr
                outseq.append(item_ctr)
                item_ctr += 1
            # else: skip unseen item
        if len(outseq) >= 2:
            new_seqs.append(outseq)
    return new_seqs

tra_seqs = reindex_session(tra_sess, train_mode=True)
tes_seqs = reindex_session(tes_sess, train_mode=False)

# Generate (sequence, label) pairs for last-item prediction
def process_seqs_no(iseqs):
    out_seqs, labs = [], []
    for seq in iseqs:
        out_seqs.append(seq[:-1])  # input session without last item
        labs.append(seq[-1])       # label is the last item
    return out_seqs, labs

tr_seqs, tr_labs = process_seqs_no(tra_seqs)
te_seqs, te_labs = process_seqs_no(tes_seqs)

print('train sequence:', tr_seqs[:5])
print('train lab:', tr_labs[:5])

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

# === Stats ===
all_interactions = sum(len(seq) for seq in tra_seqs + tes_seqs)
print('#interactions:', all_interactions)
print('#sessions:', len(tra_seqs) + len(tes_seqs))
print('#items:', len(item_dict))
print('sequence average length:', all_interactions / (len(tra_seqs) + len(tes_seqs)))

# === Save ===
output_path = "datasets/Amazon_grocery_2018/"
os.makedirs(output_path, exist_ok=True)
pickle.dump(tra, open(os.path.join(output_path, "train.txt"), 'wb'))
pickle.dump(tes, open(os.path.join(output_path, "test.txt"), 'wb'))

print("dataset:", datasets_name)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("done")
