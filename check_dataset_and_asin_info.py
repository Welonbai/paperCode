# check_dataset_and_asin_info.py
import os
import sys
import argparse
import pickle
from collections import Counter

def sniff_is_pickle(path):
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        # Pickle protocols typically start with 0x80
        return head[:1] == b"\x80"
    except Exception:
        return False

def load_train_test(path):
    """Load (seqs, labels) from pickle OR parse flat integer IDs from text."""
    if sniff_is_pickle(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # expected (seqs, labels)
        if isinstance(obj, tuple) and len(obj) == 2:
            seqs, labs = obj
            # Flatten item IDs from sequences + labels
            ids = []
            for s in seqs:
                if isinstance(s, (list, tuple)):
                    ids.extend(int(x) for x in s)
            ids.extend(int(x) for x in labs)
            return ids, (seqs, labs), "pickle"
        else:
            # Fallback: treat as a pickled list of sequences
            if isinstance(obj, list):
                ids = []
                for s in obj:
                    if isinstance(s, (list, tuple)):
                        ids.extend(int(x) for x in s)
                    else:
                        try:
                            ids.append(int(s))
                        except Exception:
                            pass
                return ids, (obj, None), "pickle(list)"
            raise ValueError(f"Unsupported pickle structure in {path}")
    else:
        # Read as text of ints (IDs) per line (space/comma/tab separated)
        ids = []
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                parts = line.replace(",", " ").replace("\t", " ").split()
                for p in parts:
                    try:
                        ids.append(int(p))
                    except Exception:
                        pass
        return ids, (None, None), "text"

def pretty_stats(name, ids):
    s = set(ids)
    if not s:
        print(f"\n[{name}] No IDs found.")
        return s
    min_id, max_id = min(s), max(s)
    contiguous_0 = (min_id == 0 and len(s) == max_id + 1)
    contiguous_1 = (min_id == 1 and len(s) == max_id)
    print(f"\n[{name}]")
    print(f"  Unique item IDs: {len(s)}")
    print(f"  Min ID: {min_id}")
    print(f"  Max ID: {max_id}")
    print(f"  0-based contiguous: {contiguous_0}")
    print(f"  1-based contiguous: {contiguous_1}")
    cnt = Counter(ids)
    print(f"  Top-5 most frequent IDs: {cnt.most_common(5)}")
    return s

def load_id_sets_from_id_asin_info(path):
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    ids = set()
    if isinstance(obj, list):
        # list of dicts or tuples
        for e in obj:
            if isinstance(e, dict) and "id" in e:
                try: ids.add(int(e["id"]))
                except: pass
            elif isinstance(e, (list, tuple)) and e:
                try: ids.add(int(e[0]))
                except: pass
    elif isinstance(obj, dict):
        # could be {id: {...}} or {asin: {..., 'id': id}}
        # if keys are ids:
        all_int_keys = True
        for k in obj.keys():
            try:
                int(k)
            except Exception:
                all_int_keys = False
                break
        if all_int_keys:
            for k in obj.keys():
                try: ids.add(int(k))
                except: pass
        else:
            # values contain 'id'
            for v in obj.values():
                if isinstance(v, dict) and "id" in v:
                    try: ids.add(int(v["id"]))
                    except: pass
    return ids

def load_id_sets_from_is_asin_info(path):
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    ids = set()
    if isinstance(obj, dict):
        # usually {asin: {... 'id': int, ...}}
        for v in obj.values():
            if isinstance(v, dict) and "id" in v:
                try: ids.add(int(v["id"]))
                except: pass
    elif isinstance(obj, list):
        # list of dicts possibly containing 'id'
        for e in obj:
            if isinstance(e, dict) and "id" in e:
                try: ids.add(int(e["id"]))
                except: pass
    return ids

def suggest_num_node(all_ids):
    if not all_ids:
        return None, "No IDs detected."
    min_id, max_id = min(all_ids), max(all_ids)
    contiguous_0 = (min_id == 0 and len(all_ids) == max_id + 1)
    contiguous_1 = (min_id == 1 and len(all_ids) == max_id)
    recommendation = max_id + 1
    note = []
    if contiguous_0:
        note.append("IDs appear 0-based contiguous.")
    elif contiguous_1:
        note.append("IDs appear 1-based contiguous.")
    else:
        note.append("IDs appear non-contiguous.")
    note.append("Use num_node = max(item_id) + 1 (safe for embedding/table size).")
    return recommendation, " ".join(note)

def main():
    ap = argparse.ArgumentParser(description="Verify train/test IDs vs asin mappings and suggest num_node.")
    ap.add_argument("--dataset_dir", default="datasets/Amazon_cellPhone_2018", help="Dataset folder.")
    ap.add_argument("--train_file", default="train.txt", help="Train file (pickle or text).")
    ap.add_argument("--test_file", default="test.txt", help="Test file (pickle or text).")
    ap.add_argument("--mapping_dir", default="mapping", help="Mapping folder under dataset_dir.")
    ap.add_argument("--save_mismatches", action="store_true", help="Save mismatch ID lists to files.")
    args = ap.parse_args()

    ds_dir = args.dataset_dir
    train_path = os.path.join(ds_dir, args.train_file)
    test_path  = os.path.join(ds_dir, args.test_file)
    map_dir   = os.path.join(ds_dir, args.mapping_dir)

    print("=== Loading train/test ===")
    train_ids, (tr_seqs, tr_labs), tkind = load_train_test(train_path)
    test_ids,  (te_seqs, te_labs), tk2   = load_train_test(test_path)
    print(f"[INFO] Detected train format: {tkind}, test format: {tk2}")

    all_ids = train_ids + test_ids
    train_set = pretty_stats("TRAIN", train_ids)
    test_set  = pretty_stats("TEST", test_ids)
    all_set   = pretty_stats("ALL (train ∪ test)", all_ids)

    # Basic sanity checks for pickle format
    if tr_seqs is not None and tr_labs is not None:
        assert len(tr_seqs) == len(tr_labs), "[ERR] Train seqs and labels length mismatch."
        if te_seqs is not None and te_labs is not None:
            assert len(te_seqs) == len(te_labs), "[ERR] Test seqs and labels length mismatch."
        # verify labels appear in all_set
        missing_train_labels = [y for y in tr_labs if y not in train_set]
        if missing_train_labels:
            print(f"[WARN] {len(missing_train_labels)} train labels not found in train ID set (unexpected).")
        if te_labs is not None:
            missing_test_labels = [y for y in te_labs if y not in all_set]
            if missing_test_labels:
                print(f"[WARN] {len(missing_test_labels)} test labels not found in global ID set (unexpected).")

    print("\n=== Loading mappings ===")
    id_asin_info_pkl = os.path.join(map_dir, "id_asin_info.pkl")
    is_asin_info_pkl = os.path.join(map_dir, "is_asin_info.pkl")

    ids_from_id_asin = load_id_sets_from_id_asin_info(id_asin_info_pkl)
    ids_from_is_asin = load_id_sets_from_is_asin_info(is_asin_info_pkl)
    mapping_ids = set()
    if ids_from_id_asin:
        print(f"[OK] IDs from id_asin_info.pkl: {len(ids_from_id_asin)} "
              f"(min={min(ids_from_id_asin)}, max={max(ids_from_id_asin)})")
        mapping_ids |= ids_from_id_asin
    else:
        print("[WARN] Could not extract IDs from id_asin_info.pkl (missing or empty).")

    if ids_from_is_asin:
        print(f"[OK] IDs from is_asin_info.pkl: {len(ids_from_is_asin)} "
              f"(min={min(ids_from_is_asin)}, max={max(ids_from_is_asin)})")
        mapping_ids |= ids_from_is_asin
    else:
        print("[INFO] is_asin_info.pkl not present or contains no IDs (ok).")

    if not mapping_ids:
        print("\n[ERROR] No mapping IDs found. Skipping cross-checks.")
    else:
        print(f"\n[MAPPING] Combined unique IDs from mappings: {len(mapping_ids)}")
        print(f"  Min mapping ID: {min(mapping_ids)}  Max mapping ID: {max(mapping_ids)}")

        print("\n=== Cross-checks (DATA ↔ MAPPING) ===")
        data_not_in_map = all_set - mapping_ids
        map_not_in_data = mapping_ids - all_set

        print(f"IDs in DATA but NOT in MAPPING: {len(data_not_in_map)}")
        if len(data_not_in_map) <= 20:
            print(f"  {sorted(list(data_not_in_map))}")
        else:
            print("  (too many to print; use --save_mismatches to export)")

        print(f"IDs in MAPPING but NOT in DATA: {len(map_not_in_data)}")
        if len(map_not_in_data) <= 20:
            print(f"  {sorted(list(map_not_in_data))}")
        else:
            print("  (too many to print; use --save_mismatches to export)")

        if args.save_mismatches:
            out1 = os.path.join(ds_dir, "ids_in_data_not_in_mapping.txt")
            out2 = os.path.join(ds_dir, "ids_in_mapping_not_in_data.txt")
            with open(out1, "w", encoding="utf-8") as f:
                for x in sorted(list(data_not_in_map)):
                    f.write(f"{x}\n")
            with open(out2, "w", encoding="utf-8") as f:
                for x in sorted(list(map_not_in_data)):
                    f.write(f"{x}\n")
            print(f"[SAVED] Mismatch lists -> {out1}, {out2}")

    print("\n=== num_node Recommendation ===")
    recommendation, note = suggest_num_node(all_set)
    if recommendation is None:
        print("Could not suggest num_node (no IDs found).")
    else:
        print(f"Recommended num_node: {recommendation}")
        print(f"Reason: {note}")
        max_id = max(all_set) if all_set else None
        if max_id is not None and max_id >= recommendation:
            print("[WARN] max(item_id) >= num_node recommendation — increase num_node!")
        else:
            print("[OK] All dataset IDs fit within [0, num_node-1] if 0-based indexing is used.")

    print("\nDone.")

if __name__ == "__main__":
    sys.exit(main())
