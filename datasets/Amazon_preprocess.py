import os
import time
import pickle
import pandas as pd
import json

print("Starting @ %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("Loading dataset...")

datasets_name = 'Amazon_cellPhone_2018/Cell_Phones_and_Accessories'
meta_filename = 'meta_Cell_Phones_and_Accessories.json'
data_path = os.path.join('datasets', datasets_name + '.json')
meta_path = os.path.join('datasets/Amazon_cellPhone_2018', meta_filename)
output_path = "datasets/Amazon_cellPhone_2018/"
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

def load_interaction_subset(path, valid_asins):
    data = []
    for entry in parse(path):
        asin = entry.get('asin')
        if asin not in valid_asins:
            continue
        reviewer = entry.get('reviewerID')
        timestamp = entry.get('unixReviewTime')
        if reviewer and timestamp:
            data.append({
                "reviewerID": reviewer,
                "asin": asin,
                "unixReviewTime": timestamp
            })
    return pd.DataFrame(data)

# === Step 1: Load metadata and filter valid items with image ===
print("Loading metadata...")
valid_items_with_images_title_category = set()
asin_to_info = {}
for entry in parse(meta_path):
    asin = entry.get('asin')
    images = entry.get('imageURLHighRes', [])
    title = entry.get('title', '').strip()
    categories = entry.get('categories') or entry.get('category')

    if not asin:
        continue

    if (
        isinstance(images, list) and len(images) > 0 and
        isinstance(title, str) and title != '' and
        isinstance(categories, list) and len(categories) > 0
    ):
        valid_items_with_images_title_category.add(asin)
        asin_to_info[asin] = {
            "url": images[0],
            "title": title,
            "category": categories
        }

# Load review data ===
print("Loading review data...")
# Replace old getDF + df filter with this:
interaction = load_interaction_subset(data_path, valid_items_with_images_title_category)
# # Keep only interactions with items that have images ===
# valid_items_with_images_title_category_df = pd.DataFrame({'asin': list(valid_items_with_images_title_category)})
# interaction = pd.merge(interaction, valid_items_with_images_title_category_df, how='inner', on='asin')
# print(f"Total interactions after image filter: {len(interaction)}")
# print(f"Interactions removed due to missing images: {df.shape[0] - len(interaction)}")
print(f"✅ Interactions loaded: {len(interaction)}")
print(f"✅ Unique sessions: {interaction['reviewerID'].nunique()}, unique items: {interaction['asin'].nunique()}")

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

# === Step 4: Remap sessionID ===
interaction = interaction.rename(columns={'reviewerID': 'sessionID', 'asin': 'itemID', 'unixReviewTime': 'time'})
session2id = {sid: idx + 1 for idx, sid in enumerate(interaction['sessionID'].unique())}
interaction['sessionID'] = interaction['sessionID'].map(session2id)

# === Step 5: Filter items with at least 10 interactions ===
item_counts = interaction['itemID'].value_counts()
items_to_keep = item_counts[item_counts >= 10].index
interaction = interaction[interaction['itemID'].isin(items_to_keep)]

# Remap itemID after filtering
item2id = {iid: idx + 1 for idx, iid in enumerate(interaction['itemID'].unique())}
interaction['itemID'] = interaction['itemID'].map(item2id)

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

# Rebuild old_id_to_asin using final mapping only
old_id_to_asin = {v: k for k, v in item2id.items()}
id_asin_info = []
for old_id, new_id in old2new_item.items():
    asin = old_id_to_asin.get(old_id)
    if asin is not None and asin in asin_to_info:
        info = asin_to_info[asin]
        id_asin_info.append({
            "id": new_id,
            "asin": asin,
            "url": info["url"],
            "title": info["title"],
            "category": info["category"]
        })

print(f"🔎 Max item ID in id_asin_info: {max([entry['id'] for entry in id_asin_info])}")
# Save pickle 
with open(os.path.join(mapping_dir, "id_asin_info.pkl"), 'wb') as f:
    pickle.dump(id_asin_info, f)
print(f"✅ Saved id_asin_info.pkl to: {mapping_dir}")
# Also save CSV
df_info = pd.DataFrame(id_asin_info)
df_info.to_csv(os.path.join(mapping_dir, "id_asin_info.csv"), index=False)
print(f"✅ Saved id_asin_info.csv to: {mapping_dir}")


max_train_id = max([max(seq) for seq in tr_seqs if len(seq) > 0] + tr_labs)
print(f"🔎 Max train item ID (after remap): {max_train_id}")
print(f"🔎 Number of unique items in train: {len(set(i for seq in tr_seqs for i in seq))}")
max_test_id = max([max(seq) for seq in te_seqs if len(seq) > 0] + te_labs)
print(f"🔎 Max test item ID (after remap): {max_test_id}")
print(f"🔎 Number of unique items in test: {len(set(i for seq in te_seqs for i in seq))}")
# === Save train/test splits ===

# === Final Save ===
print("Saving train/test splits...")
pickle.dump((tr_seqs, tr_labs), open(os.path.join(output_path, "train.txt"), 'wb'))
pickle.dump((te_seqs, te_labs), open(os.path.join(output_path, "test.txt"), 'wb'))

print("✅ All done.")
print("Train samples:", len(tr_seqs), "Test samples:", len(te_seqs))
