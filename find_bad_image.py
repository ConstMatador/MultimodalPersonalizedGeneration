import os
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_paths = {
    "train": "/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_train_with_farthest.csv",
    "val":   "/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_val_with_farthest.csv",
    "test":  "/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_test_with_farthest.csv",
}
poster_dir = "/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters"
report_csv = "/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/bad_images_report_fast.csv"

def check_one_movie(movie_id):
    img_path = os.path.join(poster_dir, f"{movie_id}.jpg")
    if not os.path.exists(img_path):
        return movie_id, img_path, "missing_file"
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            im.convert("RGB")
        return movie_id, img_path, ""   # ok
    except Exception as e:
        return movie_id, img_path, repr(e)

# 1) 收集所有去重 movie_id
all_movie_ids = set()
for split, p in csv_paths.items():
    df = pd.read_csv(p, usecols=["future_pos"])
    all_movie_ids.update(pd.to_numeric(df["future_pos"], errors="coerce").dropna().astype(int).tolist())

all_movie_ids = sorted(all_movie_ids)
print(f"unique movie_ids: {len(all_movie_ids)}")

# 2) 并行检查
bad_map = {}  # movie_id -> (path, err)
max_workers = min(32, (os.cpu_count() or 8) * 2)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(check_one_movie, mid) for mid in all_movie_ids]
    for i, fut in enumerate(as_completed(futures), 1):
        mid, path, err = fut.result()
        if err:
            bad_map[mid] = (path, err)
        if i % 1000 == 0:
            print(f"checked {i}/{len(all_movie_ids)}")

print(f"bad unique images: {len(bad_map)}")

# 3) 回填到每个 split 的行级报告
rows = []
for split, p in csv_paths.items():
    df = pd.read_csv(p, usecols=["future_pos"])
    movie_ids = pd.to_numeric(df["future_pos"], errors="coerce")
    for idx, v in enumerate(movie_ids):
        if pd.isna(v):
            rows.append([split, p, idx, None, "", "invalid_movie_id"])
            continue
        mid = int(v)
        if mid in bad_map:
            path, err = bad_map[mid]
            rows.append([split, p, idx, mid, path, err])

bad_df = pd.DataFrame(rows, columns=["split", "csv_path", "row_idx", "movie_id", "image_path", "error"])
bad_df.to_csv(report_csv, index=False, encoding="utf-8-sig")

print(f"row-level bad samples: {len(bad_df)}")
print(f"saved: {report_csv}")
print(bad_df.head(20).to_string(index=False))
