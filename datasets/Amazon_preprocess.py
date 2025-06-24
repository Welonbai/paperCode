import os
import time
import pickle
import pandas as pd
import json

print("Starting @ %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("Loading dataset...")

datasets_name = 'Amazon_grocery_2018/Grocery_and_Gourmet_Food'
meta_filename = 'meta_Grocery_and_Gourmet_Food.json'
data_path = os.path.join('datasets', datasets_name + '.json')
meta_path = os.path.join('datasets/Amazon_grocery_2018', meta_filename)
output_path = "datasets/Amazon_grocery_2018/"
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

# === Step 1: Load metadata and filter valid items with image ===
print("Loading metadata...")
valid_items_with_images = set()
for entry in parse(meta_path):
    if 'asin' in entry and 'imageURLHighRes' in entry:
        if isinstance(entry['imageURLHighRes'], list) and len(entry['imageURLHighRes']) > 0:
            valid_items_with_images.add(entry['asin'])

print(f"✅ Valid items with image: {len(valid_items_with_images)}")
pickle.dump(valid_items_with_images, open(os.path.join(output_path, 'items_with_image.pkl'), 'wb'))

# Load review data ===
print("Loading review data...")
df = getDF(data_path)
interaction = df[['reviewerID', 'asin', 'unixReviewTime']].dropna()
print(f"Total interactions before image filter: {len(interaction)}")

# Keep only interactions with items that have images ===
valid_items_with_images_df = pd.DataFrame({'asin': list(valid_items_with_images)})
interaction = pd.merge(interaction, valid_items_with_images_df, how='inner', on='asin')
print(f"Total interactions after image filter: {len(interaction)}")
print(f"Interactions removed due to missing images: {df.shape[0] - len(interaction)}")

# Drop sessions with ≤ 1 interaction ===
session_counts = interaction['reviewerID'].value_counts()
sessions_to_keep = session_counts[session_counts > 1].index
interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]
print(f"Remaining sessions after filtering single-interaction sessions: {len(sessions_to_keep)}")

# Filter sessions to only include interactions within 24h ===
session_min_time = interaction.groupby('reviewerID')['unixReviewTime'].transform('min')
interaction['time_interval'] = interaction['unixReviewTime'] - session_min_time
interaction = interaction[interaction['time_interval'] < time_interval]

# Again drop sessions that now have only one interaction after time filter
session_counts = interaction['reviewerID'].value_counts()
sessions_to_keep = session_counts[session_counts > 1].index
interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]

print("✅ Interaction filtering complete after merging with image metadata.")

# === Step 4: Remap sessionID and itemID ===
interaction = interaction.rename(columns={'reviewerID': 'sessionID', 'asin': 'itemID', 'unixReviewTime': 'time'})
session2id = {sid: idx + 1 for idx, sid in enumerate(interaction['sessionID'].unique())}
item2id = {iid: idx + 1 for idx, iid in enumerate(interaction['itemID'].unique())}
interaction['sessionID'] = interaction['sessionID'].map(session2id)
interaction['itemID'] = interaction['itemID'].map(item2id)

# === Step 5: Filter items with at least 10 interactions ===
item_counts = interaction['itemID'].value_counts()
items_to_keep = item_counts[item_counts >= 10].index
interaction = interaction[interaction['itemID'].isin(items_to_keep)]

# === Step 6: Group sessions ===
interaction = interaction.sort_values(['sessionID', 'time'])
sessions = interaction.groupby('sessionID')['itemID'].apply(list).to_dict()

# === CoHHN-style session length statistics ===
session_lengths = [len(seq) for seq in sessions.values()]
avg_session_len = sum(session_lengths) / len(session_lengths)
print(f"✅ CoHHN-style session length: {avg_session_len:.2f}")


# === Step 7: Train/test split (90/10) ===
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

# === Step 8: Generate (sequence, label) pairs from train/test ===
def process_seqs_no(iseqs):
    out_seqs, labs = [], []
    for seq in iseqs:
        if len(seq) < 2:
            continue
        input_seq = seq[:-1]
        label = seq[-1]
        out_seqs.append(input_seq)
        labs.append(label)
    return out_seqs, labs

tr_seqs_raw, tr_labs_raw = process_seqs_no(list(tra_sess.values()))
te_seqs_raw, te_labs_raw = process_seqs_no(list(tes_sess.values()))

# === Step 9: Densely remap items based only on training set ===
print("Original max train label:", max(tr_labs_raw))
print("Original number of unique train labels:", len(set(tr_labs_raw)))

all_train_items = set()
for seq in tr_seqs_raw:
    all_train_items.update(seq)
all_train_items.update(tr_labs_raw)

old2new_item = {old_id: new_id for new_id, old_id in enumerate(sorted(all_train_items), start=1)}
print("Dense new id mapping completed. New number of classes:", len(old2new_item))

# Remap train
tr_seqs = [[old2new_item[i] for i in seq] for seq in tr_seqs_raw]
tr_labs = [old2new_item[l] for l in tr_labs_raw]

# Remap test, filter out unknown items
te_seqs, te_labs = [], []
for seq, label in zip(te_seqs_raw, te_labs_raw):
    if all(i in old2new_item for i in seq + [label]):
        te_seqs.append([old2new_item[i] for i in seq])
        te_labs.append(old2new_item[label])


# # === Step 8: Re-index items and prepare (sequence, label) pairs ===
# item_dict = {}
# item_ctr = 1

# def reindex_session(sess_dict, train_mode=True):
#     global item_ctr
#     new_seqs = []
#     for s in sess_dict:
#         outseq = []
#         for i in sess_dict[s]:
#             if i in item_dict:
#                 outseq.append(item_dict[i])
#             elif train_mode:
#                 item_dict[i] = item_ctr
#                 outseq.append(item_ctr)
#                 item_ctr += 1
#         if len(outseq) >= 2:
#             new_seqs.append(outseq)
#     return new_seqs

# tra_seqs = reindex_session(tra_sess, train_mode=True)
# tes_seqs = reindex_session(tes_sess, train_mode=False)

# def process_seqs_no(iseqs):
#     out_seqs, labs = [], []
#     for seq in iseqs:
#         if len(seq) < 2:
#             continue
#         input_seq = seq[:-1]
#         label = seq[-1]
#         if label <= len(item_dict):
#             out_seqs.append(input_seq)
#             labs.append(label)
#     return out_seqs, labs

# tr_seqs, tr_labs = process_seqs_no(tra_seqs)
# te_seqs, te_labs = process_seqs_no(tes_seqs)

# # === Step 9: Densely re-map items ===
# all_train_items = set()
# for seq in tr_seqs:
#     all_train_items.update(seq)
# all_train_items.update(tr_labs)

# old2new_item = {}
# new_id = 1
# for old_id in sorted(all_train_items):
#     old2new_item[old_id] = new_id
#     new_id += 1

# tr_seqs = [[old2new_item[i] for i in seq] for seq in tr_seqs]
# tr_labs = [old2new_item[l] for l in tr_labs]

# te_seqs_remapped, te_labs_remapped = [], []
# for seq, label in zip(te_seqs, te_labs):
#     if all(i in old2new_item for i in seq + [label]):
#         te_seqs_remapped.append([old2new_item[i] for i in seq])
#         te_labs_remapped.append(old2new_item[label])

# te_seqs, te_labs = te_seqs_remapped, te_labs_remapped

# === Show sample output ===
print("After remap, train samples:", len(tr_seqs), ", test samples:", len(te_seqs))
print('train sequence:', tr_seqs[:5])
print('train lab:', tr_labs[:5])

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

# === Stats ===
all_interactions = sum(len(seq) for seq in tr_seqs + te_seqs)
print('#interactions:', all_interactions)
print('#sessions:', len(tr_seqs) + len(te_seqs))
print('#items:', len(set(i for seq in tr_seqs for i in seq)))
# print('sequence average length:', all_interactions / (len(tra_seqs) + len(tes_seqs)))

# === Final Save ===
print("Saving train/test splits...")
pickle.dump((tr_seqs, tr_labs), open(os.path.join(output_path, "train.txt"), 'wb'))
pickle.dump((te_seqs, te_labs), open(os.path.join(output_path, "test.txt"), 'wb'))

print("✅ All done.")
print("Train samples:", len(tr_seqs), "Test samples:", len(te_seqs))
