import os
import ast
import random
import argparse
import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import lpips
from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from transformers import logging as hf_logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*")
warnings.filterwarnings("ignore", message="`upcast_vae` is deprecated.*")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_embedding_str(s):
    if isinstance(s, list):
        return np.array(s, dtype=np.float32)
    return np.array(ast.literal_eval(s), dtype=np.float32)


def load_summary_map(summary_csv):
    df = pd.read_csv(summary_csv)
    df["movieId"] = df["movieId"].astype(int)
    return dict(zip(df["movieId"], df["summary"].fillna("").astype(str)))


def clean_and_truncate_summary(text: str, max_words: int = 35) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("<|endoftext|>", " ")
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace("!", " ")
    text = " ".join(text.split())  # normalize spaces
    words = text.split(" ")
    return " ".join(words[:max_words])


def image_to_tensor_01(pil_img, size=512):
    pil_img = pil_img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)  # [3,H,W], [0,1]


# -----------------------------
# Dataset
# -----------------------------
class UserPosterDataset(Dataset):
    def __init__(self, csv_path, summary_map, poster_dir, image_size=512, max_summary_words=35):
        self.df = pd.read_csv(csv_path)
        self.summary_map = summary_map
        self.poster_dir = poster_dir
        self.image_size = image_size
        self.max_summary_words = max_summary_words

        needed = ["future_pos", "embedding", "farthest_embedding"]
        for c in needed:
            if c not in self.df.columns:
                raise ValueError(f"{csv_path} missing column: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        movie_id = int(row["future_pos"])
        emb = parse_embedding_str(row["embedding"])
        far_emb = parse_embedding_str(row["farthest_embedding"])

        summary_raw = self.summary_map.get(movie_id, "")
        summary = clean_and_truncate_summary(summary_raw, max_words=self.max_summary_words)
        prompt = f"Movie poster. {summary}" if summary else "Movie poster."

        poster_path = os.path.join(self.poster_dir, f"{movie_id}.jpg")
        if not os.path.exists(poster_path):
            raise FileNotFoundError(f"Poster not found: {poster_path}")

        real_img = Image.open(poster_path)
        real_img = image_to_tensor_01(real_img, self.image_size)

        return {
            "movie_id": movie_id,
            "prompt": prompt,
            "user_emb": torch.tensor(emb, dtype=torch.float32),
            "far_emb": torch.tensor(far_emb, dtype=torch.float32),
            "real_img": real_img,
        }


def collate_fn(batch):
    return {
        "movie_id": [b["movie_id"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "user_emb": torch.stack([b["user_emb"] for b in batch], dim=0),
        "far_emb": torch.stack([b["far_emb"] for b in batch], dim=0),
        "real_img": torch.stack([b["real_img"] for b in batch], dim=0),
    }


# -----------------------------
# Adapter
# -----------------------------
class UserAdapter(nn.Module):
    def __init__(self, in_dim=256, hidden=1024, out_dim=2048, n_user_tokens=4, dropout=0.1):
        super().__init__()
        self.n_user_tokens = n_user_tokens
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_user_tokens * out_dim)
        )

    def forward(self, x):
        y = self.mlp(x)
        y = y.view(x.size(0), self.n_user_tokens, self.out_dim)
        return y


# -----------------------------
# Core Model Wrapper
# -----------------------------
class SDXLPosterPersonalizer(nn.Module):
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", n_user_tokens=4):
        super().__init__()
        self.device = device

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            variant="fp16" if "cuda" in device else None,
            use_safetensors=True
        ).to(device)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)

        # 与 SDXL prompt_embeds 对齐（关键）
        self.hidden_dim = self.pipe.unet.config.cross_attention_dim

        self.adapter = UserAdapter(
            in_dim=256, hidden=1024, out_dim=self.hidden_dim, n_user_tokens=n_user_tokens
        ).to(device)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor)

    @torch.no_grad()
    def encode_prompt(self, prompts):
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )
        return prompt_embeds, pooled_prompt_embeds

    def build_cond(self, prompts, user_emb_256):
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompts)
        user_tokens = self.adapter(user_emb_256)
        cond = torch.cat([user_tokens, prompt_embeds, user_tokens], dim=1)
        return cond, pooled_prompt_embeds

    def generate_images(self, cond_embeds, pooled_embeds, height=512, width=512, steps=20, guidance_scale=0.0):
        out = self.pipe(
            prompt_embeds=cond_embeds,
            pooled_prompt_embeds=pooled_embeds,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            output_type="pt"
        )
        return out.images  # [B,3,H,W], [0,1]


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    test_csv: str
    summary_csv: str
    poster_dir: str
    output_dir: str = "save/PosterGenerator"
    test_save_dir: str = "TestPosters"

    model_id: str = "/root/TOS/ZhongzhengWang/model/stable-diffusion-xl-base-1.0"
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 2
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lpips_w_real: float = 1.0
    lpips_w_far: float = 0.5
    n_user_tokens: int = 4
    gen_steps_train: int = 20
    gen_steps_eval: int = 30
    max_summary_words: int = 35
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def save_test_posters(user_imgs, movie_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(user_imgs.size(0)):
        movie_id = int(movie_ids[i])
        save_path = os.path.join(save_dir, f"{movie_id}.jpg")  # future_pos.jpg
        save_image(user_imgs[i].clamp(0, 1), save_path)


def run_one_epoch(model, loader, optimizer, lpips_fn, cfg: TrainConfig, train=True, save_test_images=False):
    model.train(train)
    total_loss, total_real, total_far, n = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc="train" if train else "eval")
    for batch in pbar:
        movie_ids = batch["movie_id"]
        prompts = batch["prompt"]
        user_emb = batch["user_emb"].to(cfg.device)
        far_emb = batch["far_emb"].to(cfg.device)
        real_img = batch["real_img"].to(cfg.device)

        cond_user, pooled_user = model.build_cond(prompts, user_emb)
        user_img = model.generate_images(
            cond_user, pooled_user,
            height=cfg.image_size, width=cfg.image_size,
            steps=cfg.gen_steps_train if train else cfg.gen_steps_eval
        )

        with torch.no_grad():
            cond_far, pooled_far = model.build_cond(prompts, far_emb)
            far_img = model.generate_images(
                cond_far, pooled_far,
                height=cfg.image_size, width=cfg.image_size,
                steps=cfg.gen_steps_train if train else cfg.gen_steps_eval
            )

        user_img_m1 = user_img * 2 - 1
        far_img_m1 = far_img * 2 - 1
        real_img_m1 = real_img * 2 - 1

        lp_real = lpips_fn(user_img_m1, real_img_m1).mean()
        lp_far = lpips_fn(user_img_m1, far_img_m1).mean()
        loss = cfg.lpips_w_real * lp_real - cfg.lpips_w_far * lp_far

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_test_images and (not train):
            save_test_posters(user_img.detach().cpu(), movie_ids, cfg.test_save_dir)

        bs = user_emb.size(0)
        total_loss += loss.item() * bs
        total_real += lp_real.item() * bs
        total_far += lp_far.item() * bs
        n += bs

        pbar.set_postfix({
            "loss": f"{total_loss / n:.4f}",
            "lp_real": f"{total_real / n:.4f}",
            "lp_far": f"{total_far / n:.4f}",
        })

    return {
        "loss": total_loss / max(n, 1),
        "lp_real": total_real / max(n, 1),
        "lp_far": total_far / max(n, 1),
    }


def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    # 关闭transformers/diffusers冗余警告（含截断提示）
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.test_save_dir, exist_ok=True)

    summary_map = load_summary_map(cfg.summary_csv)

    train_set = UserPosterDataset(cfg.train_csv, summary_map, cfg.poster_dir, cfg.image_size, cfg.max_summary_words)
    val_set = UserPosterDataset(cfg.val_csv, summary_map, cfg.poster_dir, cfg.image_size, cfg.max_summary_words)
    test_set = UserPosterDataset(cfg.test_csv, summary_map, cfg.poster_dir, cfg.image_size, cfg.max_summary_words)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    print(f"Using device: {cfg.device}")
    model = SDXLPosterPersonalizer(model_id=cfg.model_id, device=cfg.device, n_user_tokens=cfg.n_user_tokens)

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lpips_fn = lpips.LPIPS(net='vgg').to(cfg.device)
    lpips_fn.eval()

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")
        train_metrics = run_one_epoch(model, train_loader, optimizer, lpips_fn, cfg, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, lpips_fn=lpips_fn, cfg=cfg, train=False)

        print(f"[Train] loss={train_metrics['loss']:.4f}, lp_real={train_metrics['lp_real']:.4f}, lp_far={train_metrics['lp_far']:.4f}")
        print(f"[Val]   loss={val_metrics['loss']:.4f}, lp_real={val_metrics['lp_real']:.4f}, lp_far={val_metrics['lp_far']:.4f}")

        torch.save(model.adapter.state_dict(), os.path.join(cfg.output_dir, "adapter_last.pt"))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(cfg.output_dir, "adapter_best.pt")
            torch.save(model.adapter.state_dict(), best_path)
            print(f"Saved best checkpoint: {best_path}")

    best_path = os.path.join(cfg.output_dir, "adapter_best.pt")
    if os.path.exists(best_path):
        model.adapter.load_state_dict(torch.load(best_path, map_location=cfg.device))
        print(f"Loaded best checkpoint from {best_path}")

    test_metrics = run_one_epoch(
        model, test_loader, optimizer=None, lpips_fn=lpips_fn, cfg=cfg,
        train=False, save_test_images=True
    )
    print(f"[Test]  loss={test_metrics['loss']:.4f}, lp_real={test_metrics['lp_real']:.4f}, lp_far={test_metrics['lp_far']:.4f}")
    print(f"Generated test posters saved to: {cfg.test_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embedding_train_with_farthest.csv")
    parser.add_argument("--val_csv", type=str, default="/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_val_with_farthest.csv")
    parser.add_argument("--test_csv", type=str, default="/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_test_with_farthest.csv")
    parser.add_argument("--summary_csv", type=str, default="/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/summary.csv")
    parser.add_argument("--poster_dir", type=str, default="/root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters")
    parser.add_argument("--output_dir", type=str, default="save/PosterGenerator")
    parser.add_argument("--test_save_dir", type=str, default="TestPosters")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lpips_w_real", type=float, default=1.0)
    parser.add_argument("--lpips_w_far", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index, e.g. 0/1. If None, auto.")
    parser.add_argument("--max_summary_words", type=int, default=35)
    args = parser.parse_args()

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --gpu was specified.")
        device_str = f"cuda:{args.gpu}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        summary_csv=args.summary_csv,
        poster_dir=args.poster_dir,
        output_dir=args.output_dir,
        test_save_dir=args.test_save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        lpips_w_real=args.lpips_w_real,
        lpips_w_far=args.lpips_w_far,
        max_summary_words=args.max_summary_words,
        device=device_str,
    )
    main(cfg)
