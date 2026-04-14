# -*- coding: utf-8 -*-
import os
import ast
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from model.two_tower import ItemTower, UserTower, TwoTower


CN_TZ = ZoneInfo("Asia/Shanghai")


def log_step(msg):
    now_cn = datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_cn}] {msg}", flush=True)


def parse_emb_str(s):
    if isinstance(s, str):
        return np.array(ast.literal_eval(s), dtype=np.float32)
    elif isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=np.float32)
    else:
        raise ValueError(f"Bad embedding format: {type(s)}")


def parse_seq(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split("|") if x.strip() != ""]


class TwoTowerDataset(Dataset):
    def __init__(self, csv_file, user2idx, item2idx):
        self.df = pd.read_csv(csv_file)
        self.user2idx = user2idx
        self.item2idx = item2idx

        # 脏数据容错
        self.df["userId"] = pd.to_numeric(self.df["userId"], errors="coerce")
        self.df["future_pos"] = pd.to_numeric(self.df["future_pos"], errors="coerce")
        self.df["future_neg"] = pd.to_numeric(self.df["future_neg"], errors="coerce")

        before = len(self.df)
        self.df = self.df.dropna(subset=["userId", "future_pos", "future_neg"]).copy()

        keep = []
        for i, r in self.df.iterrows():
            uid = int(r["userId"])
            fp = int(r["future_pos"])
            fn = int(r["future_neg"])
            if uid in user2idx and fp in item2idx and fn in item2idx:
                keep.append(i)

        self.df = self.df.loc[keep].reset_index(drop=True)
        after = len(self.df)
        log_step(f"[Dataset:{os.path.basename(csv_file)}] raw={before}, valid={after}, dropped={before-after}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        uid = self.user2idx[int(r["userId"])]

        hpos = [self.item2idx[x] for x in parse_seq(r["history_pos"]) if x in self.item2idx]
        hneg = [self.item2idx[x] for x in parse_seq(r["history_neg"]) if x in self.item2idx]
        fpos = self.item2idx[int(r["future_pos"])]
        fneg = self.item2idx[int(r["future_neg"])]

        return {"uid": uid, "hpos": hpos, "hneg": hneg, "fpos": fpos, "fneg": fneg}


def collate_fn(batch, pad_idx=0):
    uids = torch.tensor([x["uid"] for x in batch], dtype=torch.long)
    fpos = torch.tensor([x["fpos"] for x in batch], dtype=torch.long)
    fneg = torch.tensor([x["fneg"] for x in batch], dtype=torch.long)

    max_p = max(1, max(len(x["hpos"]) for x in batch))
    max_n = max(1, max(len(x["hneg"]) for x in batch))

    hpos = torch.full((len(batch), max_p), pad_idx, dtype=torch.long)
    hneg = torch.full((len(batch), max_n), pad_idx, dtype=torch.long)
    mpos = torch.zeros((len(batch), max_p), dtype=torch.float32)
    mneg = torch.zeros((len(batch), max_n), dtype=torch.float32)

    for i, x in enumerate(batch):
        p = x["hpos"][:max_p]
        n = x["hneg"][:max_n]
        if len(p) > 0:
            hpos[i, :len(p)] = torch.tensor(p, dtype=torch.long)
            mpos[i, :len(p)] = 1.0
        if len(n) > 0:
            hneg[i, :len(n)] = torch.tensor(n, dtype=torch.long)
            mneg[i, :len(n)] = 1.0

    return uids, hpos, mpos, hneg, mneg, fpos, fneg


def load_item_embeddings(item_csv):
    df = pd.read_csv(item_csv)
    if "movieId" not in df.columns or "embedding" not in df.columns:
        raise ValueError("item_embedding.csv 必须包含 movieId, embedding")

    movie_ids = df["movieId"].astype(int).tolist()
    embs = [parse_emb_str(x) for x in df["embedding"].tolist()]
    dim = embs[0].shape[0]

    item2idx = {}
    mat = [np.zeros(dim, dtype=np.float32)]  # padding idx=0
    for mid, e in zip(movie_ids, embs):
        item2idx[mid] = len(mat)
        mat.append(e.astype(np.float32))

    mat = np.stack(mat, axis=0)
    return item2idx, mat


def build_user_map(csv_files):
    all_users = set()
    for p in csv_files:
        df = pd.read_csv(p, usecols=["userId"])
        df["userId"] = pd.to_numeric(df["userId"], errors="coerce")
        all_users.update(df["userId"].dropna().astype(int).tolist())

    users = sorted(all_users)
    user2idx = {u: i for i, u in enumerate(users)}
    idx2user = {i: u for u, i in user2idx.items()}
    return user2idx, idx2user


def build_loader(csv_path, user2idx, item2idx, batch_size, shuffle):
    ds = TwoTowerDataset(csv_path, user2idx, item2idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    return ds, dl


def forward_batch(model, batch, device):
    uids, hpos, mpos, hneg, mneg, fpos, fneg = batch
    uids = uids.to(device)
    hpos = hpos.to(device)
    mpos = mpos.to(device)
    hneg = hneg.to(device)
    mneg = mneg.to(device)
    fpos = fpos.to(device)
    fneg = fneg.to(device)

    hpos_vec = model.item_tower(hpos)
    hneg_vec = model.item_tower(hneg)
    u = model.user_tower(uids, hpos_vec, mpos, hneg_vec, mneg)

    vp = model.item_tower(fpos)
    vn = model.item_tower(fneg)

    sp = (u * vp).sum(dim=-1)
    sn = (u * vn).sum(dim=-1)
    loss = model.bpr_loss(u, vp, vn)
    return loss, sp, sn


@torch.no_grad()
def evaluate(model, loader, device, split_name="val"):
    model.eval()
    losses = []
    all_sp, all_sn = [], []

    for batch in tqdm(loader, desc=f"Eval-{split_name}"):
        loss, sp, sn = forward_batch(model, batch, device)
        losses.append(loss.item())
        all_sp.append(sp.cpu())
        all_sn.append(sn.cpu())

    if len(all_sp) == 0:
        return {"loss": np.nan, "pair_acc": np.nan, "margin": np.nan, "auc_like": np.nan}

    sp = torch.cat(all_sp, dim=0)
    sn = torch.cat(all_sn, dim=0)

    pair_acc = (sp > sn).float().mean().item()
    margin = (sp - sn).mean().item()
    auc_like = pair_acc

    return {
        "loss": float(np.mean(losses)) if len(losses) else np.nan,
        "pair_acc": pair_acc,
        "margin": margin,
        "auc_like": auc_like
    }


@torch.no_grad()
def export_user_embedding_per_row(model, ds, device, out_csv):
    model.eval()
    log_step(f"Export(per-row): start -> {out_csv}")

    rows = []
    for i in tqdm(range(len(ds)), desc=f"Export {os.path.basename(out_csv)}"):
        x = ds[i]

        uid = torch.tensor([x["uid"]], dtype=torch.long, device=device)
        hpos = torch.tensor([x["hpos"] if len(x["hpos"]) > 0 else [0]], dtype=torch.long, device=device)
        hneg = torch.tensor([x["hneg"] if len(x["hneg"]) > 0 else [0]], dtype=torch.long, device=device)

        mpos = torch.tensor([[1.0] * len(x["hpos"]) if len(x["hpos"]) > 0 else [0.0]], dtype=torch.float32, device=device)
        mneg = torch.tensor([[1.0] * len(x["hneg"]) if len(x["hneg"]) > 0 else [0.0]], dtype=torch.float32, device=device)

        hpos_vec = model.item_tower(hpos)
        hneg_vec = model.item_tower(hneg)
        ue = model.user_tower(uid, hpos_vec, mpos, hneg_vec, mneg).squeeze(0).cpu().numpy()

        r = ds.df.iloc[i]
        rows.append({
            "row_id": int(i),
            "userId": int(r["userId"]),
            "history_pos": str(r["history_pos"]),
            "history_neg": str(r["history_neg"]),
            "future_pos": int(r["future_pos"]),
            "future_neg": int(r["future_neg"]),
            "embedding": "[" + ",".join([f"{v:.6f}" for v in ue.tolist()]) + "]"
        })

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    log_step(f"Saved embeddings -> {out_csv}")


def train(args):
    # 1) 设备
    log_step("Step 1/10: 初始化设备")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，但你指定了 cuda 设备")
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    log_step(f"Using device: {device}")

    # 2) item向量
    log_step("Step 2/10: 读取 item embedding")
    item2idx, item_mat = load_item_embeddings(args.item_csv)
    log_step(f"item count(with pad)={item_mat.shape[0]}, item_dim={item_mat.shape[1]}")

    # 3) user映射（train+val+test联合构建）
    log_step("Step 3/10: 构建用户映射(来自train+val+test)")
    user2idx, idx2user = build_user_map([args.train_csv, args.val_csv, args.test_csv])
    log_step(f"user count(all splits)={len(user2idx)}")

    # 4) dataloader
    log_step("Step 4/10: 构建 train/val/test DataLoader")
    train_ds, train_dl = build_loader(args.train_csv, user2idx, item2idx, args.batch_size, shuffle=True)
    val_ds, val_dl = build_loader(args.val_csv, user2idx, item2idx, args.batch_size, shuffle=False)
    test_ds, test_dl = build_loader(args.test_csv, user2idx, item2idx, args.batch_size, shuffle=False)
    log_step(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # 5) model
    log_step("Step 5/10: 初始化模型")
    item_tower = ItemTower(item_emb_matrix=item_mat, out_dim=args.embed_dim, train_item_proj=args.train_item_proj)
    user_tower = UserTower(num_users=len(user2idx), d=args.embed_dim)
    model = TwoTower(item_tower, user_tower).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_step(f"model params total={total_params}, trainable={trainable_params}")

    # 6) optimizer
    log_step("Step 6/10: 初始化优化器")
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    log_step(f"optimizer=AdamW, lr={args.lr}")

    # 6.5) Zero-shot val（训练前）
    log_step("Step 6.5/10: 训练前 Zero-shot 验证")
    zs_val_metrics = evaluate(model, val_dl, device, split_name="val_zeroshot")
    log_step(
        f"[Zero-shot VAL] "
        f"loss={zs_val_metrics['loss']:.6f}, "
        f"pair_acc={zs_val_metrics['pair_acc']:.6f}, "
        f"margin={zs_val_metrics['margin']:.6f}, "
        f"auc_like={zs_val_metrics['auc_like']:.6f}"
    )

    # 7) 训练 + val
    log_step("Step 7/10: 训练并在val评估(按 val_pair_acc 选最佳)")
    os.makedirs(args.model_dir, exist_ok=True)
    best_path = os.path.join(args.model_dir, "best_two_tower.pt")
    last_path = os.path.join(args.model_dir, "last_two_tower.pt")

    best_val_pair_acc = -1.0
    best_epoch = -1

    for ep in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_dl, desc=f"Train {ep}/{args.epochs}"):
            loss, _, _ = forward_batch(model, batch, device)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if len(train_losses) else np.nan
        val_metrics = evaluate(model, val_dl, device, split_name="val")

        log_step(
            f"[Epoch {ep}] train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}, "
            f"val_pair_acc={val_metrics['pair_acc']:.6f}, "
            f"val_margin={val_metrics['margin']:.6f}"
        )

        # 保存 last
        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "idx2user": idx2user,
            "item2idx": item2idx,
            "embed_dim": args.embed_dim,
            "val_metrics": val_metrics
        }, last_path)

        # 保存 best
        if val_metrics["pair_acc"] > best_val_pair_acc:
            best_val_pair_acc = val_metrics["pair_acc"]
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "user2idx": user2idx,
                "idx2user": idx2user,
                "item2idx": item2idx,
                "embed_dim": args.embed_dim,
                "val_metrics": val_metrics
            }, best_path)
            log_step(f"New best model at epoch {ep}, val_pair_acc={best_val_pair_acc:.6f}")

    # 8) test
    log_step("Step 8/10: 加载best并在test评估")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_dl, device, split_name="test")
    log_step(
        f"[Best Epoch {best_epoch}] TEST -> "
        f"loss={test_metrics['loss']:.6f}, "
        f"pair_acc={test_metrics['pair_acc']:.6f}, "
        f"margin={test_metrics['margin']:.6f}, "
        f"auc_like={test_metrics['auc_like']:.6f}"
    )

    # 9) 保存指标
    log_step("Step 9/10: 保存评估指标")
    os.makedirs(args.metrics_dir, exist_ok=True)
    metrics_path = os.path.join(args.metrics_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "zeroshot_val_metrics": zs_val_metrics,
            "best_epoch": best_epoch,
            "best_val_pair_acc": best_val_pair_acc,
            "test_metrics": test_metrics
        }, f, ensure_ascii=False, indent=2)
    log_step(f"Saved metrics -> {metrics_path}")

    # 10) 导出三份 embedding
    log_step("Step 10/10: 导出 train/val/test 逐行 user embedding")
    export_user_embedding_per_row(model, train_ds, device, args.user_emb_csv_train)
    export_user_embedding_per_row(model, val_ds, device, args.user_emb_csv_val)
    export_user_embedding_per_row(model, test_ds, device, args.user_emb_csv_test)
    log_step("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--item_csv", type=str, required=True)

    parser.add_argument("--model_dir", type=str, default="./output_two_tower/model")
    parser.add_argument("--metrics_dir", type=str, default="./output_two_tower/metrics")

    parser.add_argument("--user_emb_csv_train", type=str, default="./output_two_tower/emb/user_embedding_train_per_row.csv")
    parser.add_argument("--user_emb_csv_val", type=str, default="./output_two_tower/emb/user_embedding_val_per_row.csv")
    parser.add_argument("--user_emb_csv_test", type=str, default="./output_two_tower/emb/user_embedding_test_per_row.csv")

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--train_item_proj", action="store_true")
    args = parser.parse_args()

    log_step("Program started")
    log_step(f"Args: {args}")
    train(args)
