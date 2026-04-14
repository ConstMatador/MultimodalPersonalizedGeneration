import os
import ast
import random
import argparse
import logging
import warnings
from dataclasses import dataclass

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.utils import save_image

import lpips
from diffusers import StableDiffusionXLPipeline
from transformers import logging as hf_logging
from accelerate import Accelerator


# -----------------------------
# Warnings & logs
# -----------------------------
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
    text = text.replace("<|endoftext|>", " ").replace("\n", " ").replace("\t", " ").replace("!", " ")
    text = " ".join(text.split())
    return " ".join(text.split(" ")[:max_words])


def image_to_tensor_01(pil_img, size=512):
    pil_img = pil_img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


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

        for c in ["future_pos", "embedding", "farthest_embedding"]:
            if c not in self.df.columns:
                raise ValueError(f"{csv_path} missing column: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        movie_id = int(row["future_pos"])
        emb = parse_embedding_str(row["embedding"])
        far_emb = parse_embedding_str(row["farthest_embedding"])

        summary = clean_and_truncate_summary(self.summary_map.get(movie_id, ""), self.max_summary_words)
        prompt = f"Movie poster. {summary}" if summary else "Movie poster."

        poster_path = os.path.join(self.poster_dir, f"{movie_id}.jpg")
        if not os.path.exists(poster_path):
            raise FileNotFoundError(f"Poster not found: {poster_path}")

        real_img = image_to_tensor_01(Image.open(poster_path), self.image_size)

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
        return y.view(x.size(0), self.n_user_tokens, self.out_dim)


# -----------------------------
# Core Wrapper
# -----------------------------
class SDXLPosterPersonalizer(nn.Module):
    def __init__(self, model_id, device, n_user_tokens=4):
        super().__init__()
        self.device = device

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
            variant="fp16" if "cuda" in str(device) else None,
            use_safetensors=True
        ).to(device)

        self.pipe.set_progress_bar_config(disable=True)

        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)

        self.hidden_dim = self.pipe.unet.config.cross_attention_dim
        self.adapter = UserAdapter(256, 1024, self.hidden_dim, n_user_tokens).to(device)

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

    def generate_images(self, cond_embeds, pooled_embeds, height=512, width=512, steps=20):
        out = self.pipe(
            prompt_embeds=cond_embeds,
            pooled_prompt_embeds=pooled_embeds,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=height,
            width=width,
            output_type="pt"
        )
        return out.images


# -----------------------------
# Train Config
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
    batch_size: int = 4               # per-GPU batch size
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
    seed: int = 42


def save_test_posters(user_imgs, movie_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(user_imgs.size(0)):
        save_image(user_imgs[i].clamp(0, 1), os.path.join(save_dir, f"{int(movie_ids[i])}.jpg"))


def reduce_scalar(accelerator, x):
    t = torch.tensor(float(x), device=accelerator.device)
    t = accelerator.reduce(t, reduction="sum")
    return t.item()


def run_one_epoch(model, loader, optimizer, lpips_fn, cfg, accelerator, train=True, save_test_images=False):
    model.adapter.train(train)

    total_loss = 0.0
    total_real = 0.0
    total_far = 0.0
    total_n = 0

    pbar = tqdm(loader, desc="train" if train else "eval", disable=not accelerator.is_local_main_process)

    for batch in pbar:
        movie_ids = batch["movie_id"]
        prompts = batch["prompt"]
        user_emb = batch["user_emb"].to(accelerator.device, non_blocking=True)
        far_emb = batch["far_emb"].to(accelerator.device, non_blocking=True)
        real_img = batch["real_img"].to(accelerator.device, non_blocking=True)

        # user branch
        cond_user, pooled_user = model.build_cond(prompts, user_emb)
        user_img = model.generate_images(
            cond_user, pooled_user,
            height=cfg.image_size, width=cfg.image_size,
            steps=cfg.gen_steps_train if train else cfg.gen_steps_eval
        )

        # far branch (no grad)
        with torch.no_grad():
            cond_far, pooled_far = model.build_cond(prompts, far_emb)
            far_img = model.generate_images(
                cond_far, pooled_far,
                height=cfg.image_size, width=cfg.image_size,
                steps=cfg.gen_steps_train if train else cfg.gen_steps_eval
            )

        lp_real = lpips_fn(user_img * 2 - 1, real_img * 2 - 1).mean()
        lp_far = lpips_fn(user_img * 2 - 1, far_img * 2 - 1).mean()
        loss = cfg.lpips_w_real * lp_real - cfg.lpips_w_far * lp_far

        if train:
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()

        if save_test_images and (not train) and accelerator.is_main_process:
            save_test_posters(user_img.detach().cpu(), movie_ids, cfg.test_save_dir)

        bs = user_emb.size(0)
        total_loss += loss.detach().item() * bs
        total_real += lp_real.detach().item() * bs
        total_far += lp_far.detach().item() * bs
        total_n += bs

        if accelerator.is_local_main_process:
            pbar.set_postfix({
                "loss": f"{total_loss/max(total_n,1):.4f}",
                "lp_real": f"{total_real/max(total_n,1):.4f}",
                "lp_far": f"{total_far/max(total_n,1):.4f}",
            })

    # 全进程汇总
    sum_loss = reduce_scalar(accelerator, total_loss)
    sum_real = reduce_scalar(accelerator, total_real)
    sum_far = reduce_scalar(accelerator, total_far)
    sum_n = reduce_scalar(accelerator, total_n)

    return {
        "loss": sum_loss / max(sum_n, 1.0),
        "lp_real": sum_real / max(sum_n, 1.0),
        "lp_far": sum_far / max(sum_n, 1.0),
    }


def main(cfg: TrainConfig):
    accelerator = Accelerator()
    device = accelerator.device

    seed_everything(cfg.seed + accelerator.process_index)
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.test_save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    summary_map = load_summary_map(cfg.summary_csv)

    train_set = UserPosterDataset(cfg.train_csv, summary_map, cfg.poster_dir, cfg.image_size, cfg.max_summary_words)
    val_set = UserPosterDataset(cfg.val_csv, summary_map, cfg.poster_dir, cfg.image_size, cfg.max_summary_words)
    test_set = UserPosterDataset(cfg.test_csv, summary_map, cfg.poster_dir, cfg.image_size, cfg.max_summary_words)

    train_sampler = DistributedSampler(train_set, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, sampler=val_sampler, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, sampler=test_sampler, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)

    if accelerator.is_main_process:
        print(f"Using device={device}, world_size={accelerator.num_processes}")

    model = SDXLPosterPersonalizer(cfg.model_id, device, cfg.n_user_tokens)

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    # 只把adapter交给DDP
    model.adapter, optimizer = accelerator.prepare(model.adapter, optimizer)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")

        train_metrics = run_one_epoch(model, train_loader, optimizer, lpips_fn, cfg, accelerator, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, lpips_fn=lpips_fn, cfg=cfg, accelerator=accelerator, train=False)

        if accelerator.is_main_process:
            print(f"[Train] loss={train_metrics['loss']:.4f}, lp_real={train_metrics['lp_real']:.4f}, lp_far={train_metrics['lp_far']:.4f}")
            print(f"[Val]   loss={val_metrics['loss']:.4f}, lp_real={val_metrics['lp_real']:.4f}, lp_far={val_metrics['lp_far']:.4f}")

            unwrapped = accelerator.unwrap_model(model.adapter)
            torch.save(unwrapped.state_dict(), os.path.join(cfg.output_dir, "adapter_last.pt"))
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(unwrapped.state_dict(), os.path.join(cfg.output_dir, "adapter_best.pt"))
                print("Saved best checkpoint.")

        accelerator.wait_for_everyone()

    best_ckpt = os.path.join(cfg.output_dir, "adapter_best.pt")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        accelerator.unwrap_model(model.adapter).load_state_dict(state)
        if accelerator.is_main_process:
            print(f"Loaded best checkpoint: {best_ckpt}")

    test_metrics = run_one_epoch(
        model, test_loader, optimizer=None, lpips_fn=lpips_fn, cfg=cfg,
        accelerator=accelerator, train=False, save_test_images=True
    )

    if accelerator.is_main_process:
        print(f"[Test]  loss={test_metrics['loss']:.4f}, lp_real={test_metrics['lp_real']:.4f}, lp_far={test_metrics['lp_far']:.4f}")
        print(f"Generated test posters saved to: {cfg.test_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--poster_dir", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="save/PosterGenerator")
    parser.add_argument("--test_save_dir", type=str, default="TestPosters")
    parser.add_argument("--model_id", type=str, default="/root/TOS/ZhongzhengWang/model/stable-diffusion-xl-base-1.0")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)  # 每卡batch
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lpips_w_real", type=float, default=1.0)
    parser.add_argument("--lpips_w_far", type=float, default=0.5)
    parser.add_argument("--n_user_tokens", type=int, default=4)
    parser.add_argument("--gen_steps_train", type=int, default=20)
    parser.add_argument("--gen_steps_eval", type=int, default=30)
    parser.add_argument("--max_summary_words", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        summary_csv=args.summary_csv,
        poster_dir=args.poster_dir,
        output_dir=args.output_dir,
        test_save_dir=args.test_save_dir,
        model_id=args.model_id,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        lpips_w_real=args.lpips_w_real,
        lpips_w_far=args.lpips_w_far,
        n_user_tokens=args.n_user_tokens,
        gen_steps_train=args.gen_steps_train,
        gen_steps_eval=args.gen_steps_eval,
        max_summary_words=args.max_summary_words,
        seed=args.seed,
    )
    main(cfg)
