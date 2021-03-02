import json
import os
import sys

import pandas as pd

checkbox_names = (
    ("pose", "Поза", "Поза", "Q"),
    ("pose_back", "Вид<br />сзади", "Вид сзади", "W"),
    ("pose_side", "Вид<br />сбоку", "Вид сбоку", "E"),
    ("hands_bad", "Руки<br />не вдоль<br />туловища", "Руки не вдоль туловища", "R"),
    ("cloths", "Одежда", "Одежда", "A"),
    ("transparent_sleeves", "Полупроз.<br />рукава", "Полупрозрачные рукава", "S"),
    ("clothes_mask_holes", "Дырки<br />в маске<br />одежды", "Дырки в маске одежды", "D"),
    ("wrong_segmentation", "Неверная<br />сегментация", "Неверная сегментация", "F"),
)

input_xlsx_dir = sys.argv[1]
ce_losses_ids_file = sys.argv[2]
output_file = sys.argv[3]

with open(ce_losses_ids_file, "r") as f:
    ce_losses = list(reversed(json.load(f)))

checked_ids = set()
filtered_ids = {}

for filename in sorted(os.listdir(input_xlsx_dir)):
    if not filename.lower().endswith(".xlsx"):
        print(f"Skipping {filename}")
        continue
    try:
        file_page_start = int(
            filename.replace(".xlsx", "").replace("Cloth-dataset-check-", "")
        )
    except ValueError:
        print(f"Skipping {filename}")
        continue

    print(filename)
    checked_ids_this_file = set(
        (item["id"] for item in ce_losses[file_page_start:file_page_start+1000])
    )
    checked_ids = checked_ids.union(checked_ids_this_file)

    check_df = pd.read_excel(os.path.join(input_xlsx_dir, filename))

    filtered_ids_this_file = {}
    for key, _, pose_name, _ in checkbox_names:
        filtered_set = set(check_df[pose_name].dropna().astype("int").values)

        if not filtered_set.issubset(checked_ids_this_file):
            print("These ids could not be present in this check file",
                  filtered_set.difference(checked_ids_this_file))
            filtered_set -= filtered_set.difference(checked_ids_this_file)

        filtered_ids_this_file[key] = filtered_set

        filtered_ids[key] = filtered_ids.get(key, set()).union(filtered_set)

    filtered_ids_this_file_union = {
        item for values in filtered_ids_this_file.values() for item in values
    }

filtered_ids_union = {item for values in filtered_ids.values() for item in values}
for key, values in filtered_ids.items():
    print(key, len(values))

print(len(filtered_ids_union))
print(len(checked_ids))
print(len(checked_ids - filtered_ids_union))

dataset_keys = {item["id"]: item["key"] for item in ce_losses}

dataset_keys_checked = [dataset_keys[item] for item in checked_ids - filtered_ids_union]
print("Left with:", len(dataset_keys_checked))

with open(output_file, "w") as f:
    json.dump(dataset_keys_checked, f)
