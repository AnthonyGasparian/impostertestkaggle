import os
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = "data/train"
meta_csv = "data/train.csv"

meta = pd.read_csv(meta_csv)   # columns: id, real_text_id

records = []
for _, row in meta.iterrows():
    art_id = row["id"]
    real_id = str(row["real_text_id"])
    folder = f"article_{art_id:04d}"
    article_path = os.path.join(base_dir, folder)

    with open(os.path.join(article_path, "file_1.txt"), encoding="utf-8") as f:
        text1 = f.read()
    with open(os.path.join(article_path, "file_2.txt"), encoding="utf-8") as f:
        text2 = f.read()

    records.append({"text": text1 if real_id == "1" else text2, "label": 1})
    records.append({"text": text2 if real_id == "1" else text1, "label": 0})

df = pd.DataFrame(records)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("train_processed.csv", index=False)
val_df.to_csv("val_processed.csv", index=False)