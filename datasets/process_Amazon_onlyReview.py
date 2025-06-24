import os
import time
import pickle
import pandas as pd
import json

print("Starting @ %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("Loading dataset...")

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

print(f"GetDF complete")

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

print(f"Filtering complete")

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

print(f"Grouping complete, about to split train/test")

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
        if len(seq) < 2:
            continue  # Skip sessions shorter than 2
        input_seq = seq[:-1]
        label = seq[-1]
        if label <= len(item_dict):  # Make sure label exists within train item dict
            out_seqs.append(input_seq)
            labs.append(label)
    return out_seqs, labs



tr_seqs, tr_labs = process_seqs_no(tra_seqs)
te_seqs, te_labs = process_seqs_no(tes_seqs)

# === Analyze input sequence lengths ===
short_count = 0
total_count = 0

for seq in tr_seqs:
    if len(seq) == 1:
        short_count += 1
    total_count += 1

print(f"Input sequences with length 1: {short_count}")
print(f"Total input sequences: {total_count}")
print(f"Percentage of short input sequences: {short_count / total_count * 100:.2f}%")

# === Densely Remap Items Based Only on All Training Items (inputs and labels) ===
print("Original max train label:", max(tr_labs))
print("Original number of unique train labels:", len(set(tr_labs)))

# Step 1. Collect all training items
all_train_items = set()
for seq in tr_seqs:
    all_train_items.update(seq)
all_train_items.update(tr_labs)

# Step 2. Build dense ID mapping
old2new_item = {}
new_id = 1
for old_id in sorted(all_train_items):
    old2new_item[old_id] = new_id
    new_id += 1

print("Dense new id mapping completed. New number of classes:", len(old2new_item))

# Step 3. Remap train sequences and labels
tr_seqs = [[old2new_item[i] for i in seq] for seq in tr_seqs]
tr_labs = [old2new_item[l] for l in tr_labs]

# Step 4. Remap test sequences and labels (only keep sessions where all items are known)
te_seqs_remapped = []
te_labs_remapped = []
for seq, label in zip(te_seqs, te_labs):
    if all(i in old2new_item for i in seq + [label]):
        te_seqs_remapped.append([old2new_item[i] for i in seq])
        te_labs_remapped.append(old2new_item[label])

te_seqs = te_seqs_remapped
te_labs = te_labs_remapped

print("After remap, train samples:", len(tr_seqs), ", test samples:", len(te_seqs))

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
