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
asin_to_url = {}
for entry in parse(meta_path):
    if 'asin' in entry and 'imageURLHighRes' in entry:
        if isinstance(entry['imageURLHighRes'], list) and len(entry['imageURLHighRes']) > 0:
            asin = entry['asin']
            valid_items_with_images.add(asin)
            asin_to_url[asin] = entry['imageURLHighRes'][0]  # Save first image URL

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

# === Save mapping directory ===
mapping_dir = os.path.join(output_path, "mapping")
os.makedirs(mapping_dir, exist_ok=True)

# === Save id ↔ asin ↔ url
id_asin_url = []
for asin, old_id in item2id.items():
    if old_id in old2new_item:
        new_id = old2new_item[old_id]
        if asin in asin_to_url:
            id_asin_url.append({"id": new_id, "asin": asin, "url": asin_to_url[asin]})

with open(os.path.join(mapping_dir, "id_asin_url.pkl"), 'wb') as f:
    pickle.dump(id_asin_url, f)

pd.DataFrame(id_asin_url).to_csv(os.path.join(mapping_dir, "id_asin_url.csv"), index=False)

print(f"✅ Saved id_asin_url.pkl and id_asin_url.csv to: {mapping_dir}")

# === Final Save ===
print("Saving train/test splits...")
pickle.dump((tr_seqs, tr_labs), open(os.path.join(output_path, "train.txt"), 'wb'))
pickle.dump((te_seqs, te_labs), open(os.path.join(output_path, "test.txt"), 'wb'))

print("✅ All done.")
print("Train samples:", len(tr_seqs), "Test samples:", len(te_seqs))
