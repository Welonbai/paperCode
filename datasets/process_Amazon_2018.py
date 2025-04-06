import json
import pandas as pd
import pickle

# Load data from gzip JSON file
def load_amazon_grocery(path):
    def parse(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    data = [review for review in parse(path)]
    return pd.DataFrame(data)


df = load_amazon_grocery('datasets/Amazon_grocery_2018/Grocery_and_Gourmet_Food_5.json')
print(f"🔹 Original reviews: {len(df)}")
print(f"🔹 Original unique items: {df['asin'].nunique()}")
print(f"🔹 Original unique users: {df['reviewerID'].nunique()}")

interaction = df[['reviewerID', 'asin', 'unixReviewTime']]

# Step 1: Filter items with < 10 interactions
item_counts = interaction['asin'].value_counts()
items_to_keep = item_counts[item_counts >= 10].index
removed_items = item_counts[item_counts < 10].shape[0]
interaction = interaction[interaction['asin'].isin(items_to_keep)]
print(f"✅ Items kept: {len(items_to_keep)} (removed: {removed_items})")
print(f"✅ Interactions after item filter: {len(interaction)}")

# Step 2: Filter sessions (users) with only 1 interaction
session_counts = interaction['reviewerID'].value_counts()
sessions_to_keep = session_counts[session_counts > 1].index
removed_sessions = session_counts[session_counts <= 1].shape[0]
interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]
print(f"✅ Sessions kept (len >1): {len(sessions_to_keep)} (removed: {removed_sessions})")
print(f"✅ Interactions after session filter: {len(interaction)}")

# Step 3: Filter by 24-hour window
time_interval = 60 * 60 * 24
session_min_time = interaction.groupby('reviewerID')['unixReviewTime'].transform('min')
interaction['time_interval'] = interaction['unixReviewTime'] - session_min_time
interaction = interaction[interaction['time_interval'] < time_interval]
print(f"✅ Interactions after 24h filter: {len(interaction)}")

# Re-apply session filter after time filter
session_counts = interaction['reviewerID'].value_counts()
sessions_to_keep = session_counts[session_counts > 1].index
interaction = interaction[interaction['reviewerID'].isin(sessions_to_keep)]
print(f"✅ Final sessions (len >1): {len(sessions_to_keep)}")
print(f"✅ Final interactions: {len(interaction)}")
print(f"✅ Final unique items: {interaction['asin'].nunique()}")

# Rename columns for session-based recommendation clearly
interaction.rename(columns={'reviewerID': 'sessionID', 'asin': 'itemID', 'unixReviewTime': 'timestamp'}, inplace=True)

# Map item IDs to integers (necessary for GCEGNN)
unique_items = interaction['itemID'].unique()
item_id_map = {item: idx + 1 for idx, item in enumerate(unique_items)}  # start IDs from 1
interaction['itemID'] = interaction['itemID'].map(item_id_map)

# Generate session sequences sorted by timestamp
interaction = interaction.sort_values(['sessionID', 'timestamp'])
sessions = interaction.groupby('sessionID')['itemID'].apply(list).tolist()

# Split sessions by 90/10 chronological order (CoHHN style)
timestamps = interaction.groupby('sessionID')['timestamp'].max()
sorted_sessions = list(timestamps.sort_values().index)

cutoff = int(0.9 * len(sorted_sessions))
train_sess_ids = set(sorted_sessions[:cutoff])
test_sess_ids = set(sorted_sessions[cutoff:])

session_map = interaction.groupby('sessionID')['itemID'].apply(list).to_dict()
train_sessions = [session_map[sid] for sid in sorted_sessions[:cutoff]]
test_sessions = [session_map[sid] for sid in sorted_sessions[cutoff:]]

# Prepare train and test data in GCEGNN expected format
def generate_sequence_target(sessions):
    sequences, targets = [], []
    for sess in sessions:
        for i in range(1, len(sess)):
            sequences.append(sess[:i])
            targets.append(sess[i])
    return [sequences, targets]

train_data = generate_sequence_target(train_sessions)
test_data = generate_sequence_target(test_sessions)

# Save preprocessed dataset in GCEGNN compatible format
pickle.dump(train_data, open('datasets/Amazon_grocery_2018/train.txt', 'wb'))
pickle.dump(test_data, open('datasets/Amazon_grocery_2018/test.txt', 'wb'))

# (Optional) Save item mapping for reference
# pickle.dump(item_id_map, open('datasets/Amazon_grocery_2018/item_id_map.pkl', 'wb'))

# Stats for verification
print('Number of unique sessions:', interaction['sessionID'].nunique())
print('Number of unique items:', len(item_id_map))
print('Number of interactions:', len(interaction))



# import json
# import pandas as pd
# import pickle

# def load_amazon_reviews(path):
#     data = []
#     with open(path, 'rt', encoding='utf-8') as f:
#         for line in f:
#             review = json.loads(line.strip())
#             user = review['reviewerID']
#             item = review['asin']
#             timestamp = review['unixReviewTime']
#             data.append([user, item, timestamp])
#     return pd.DataFrame(data, columns=['user', 'item', 'time'])

# reviews_df = load_amazon_reviews('datasets/Amazon_grocery_2018/Grocery_and_Gourmet_Food_5.json')
# print(f"Loaded {len(reviews_df)} reviews initially.")

# # Single-step filtering for items (occurrences ≥ 5)
# original_items_num = reviews_df['item'].nunique()
# print(f"📌 Original number of unique items: {original_items_num}")
# item_counts = reviews_df['item'].value_counts()
# items_to_keep = item_counts[item_counts >= 3].index
# reviews_df = reviews_df[reviews_df['item'].isin(items_to_keep)]
# print(f"✅ Items remaining after filtering: {len(items_to_keep)}")

# # Sort by user and time explicitly
# reviews_df.sort_values(by=['user', 'time'], inplace=True)

# session_gap = 3600  # 1 hour gap
# sessions, labels = [], []

# for user, user_group in reviews_df.groupby('user'):
#     session, last_time = [], None
#     for _, row in user_group.iterrows():
#         cur_time = row['time']
#         if last_time is None or cur_time - last_time <= session_gap:
#             session.append(row['item'])
#         else:
#             if len(session) >= 2:
#                 sessions.append(session[:-1])
#                 labels.append(session[-1])
#             session = [row['item']]
#         last_time = cur_time
#     # explicitly handle last session
#     if len(session) >= 2:
#         sessions.append(session[:-1])
#         labels.append(session[-1])

# print(f'✅ Total sessions extracted: {len(sessions)}')

# # Remap items explicitly
# unique_items = sorted(set(item for session in sessions for item in session) | set(labels))
# item2id = {item: idx for idx, item in enumerate(unique_items)}

# sessions_mapped = [[item2id[i] for i in s] for s in sessions]
# labels_mapped = [item2id[lbl] for lbl in labels]

# print(f"✅ Final num_node (unique items): {len(unique_items)}")

# # Save explicitly to pickle file
# with open('amazon_grocery_train.txt', 'wb') as f:
#     pickle.dump((sessions_mapped, labels_mapped), f)

# print("✅ Final Session data saved as amazon_grocery_train.txt")
