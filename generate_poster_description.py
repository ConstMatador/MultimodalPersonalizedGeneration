import os
import re
import math
import time
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*generation flags are not valid.*"
)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def extract_title_from_info(info_text: str) -> str:
    if not isinstance(info_text, str) or not info_text.strip():
        return "Unknown Title"

    text = info_text.strip()

    m = re.search(r"(?i)\btitle\s*[:：]\s*([^\n|;,]+)", text)
    if m:
        return m.group(1).strip()

    first_part = re.split(r"[。\n|;]", text)[0].strip()
    if first_part:
        first_part = re.sub(r"(?i)^\s*movie\s*[:：]\s*", "", first_part).strip()
        if len(first_part) > 80:
            first_part = first_part[:80].rstrip()
        return first_part

    return "Unknown Title"


def build_prompt(title: str) -> str:
    return (
        "Describe the movie poster in English. "
        f"The movie title is: {title}. "
        "Requirements: start explicitly with the movie title, "
        "then describe key visual elements; one sentence only; max 50 words; "
        "output final answer only, no analysis."
    )


def load_model_and_processor(model_path: str, dtype: str = "bfloat16", max_pixels: int = 640 * 28 * 28):
    if dtype == "bfloat16":
        dt = torch.bfloat16
    elif dtype == "float16":
        dt = torch.float16
    else:
        dt = torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dt,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels
    )

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    model.eval()
    return model, processor


def postprocess_text(gen_text: str, title: str, max_words: int = 50) -> str:
    s = (gen_text or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)

    if not s:
        s = f"{title}: Poster with cinematic composition and dramatic visual style."

    if not s.lower().startswith(title.lower()):
        s = f"{title}: {s}"

    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",.;:") + "."

    return s


@torch.no_grad()
def generate_batch(model, processor, image_paths, titles, max_new_tokens=48):
    images = [Image.open(p).convert("RGB") for p in image_paths]

    messages = []
    for title in titles:
        prompt = build_prompt(title)
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ],
            }
        ])

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device, non_blocking=True)

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        repetition_penalty=1.05
    )

    input_len = inputs["input_ids"].shape[1]
    gen_texts = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    cleaned = [postprocess_text(t, title) for t, title in zip(gen_texts, titles)]
    return cleaned


def main(args):
    df = pd.read_csv(args.input_csv)

    if "movieId" not in df.columns or "info" not in df.columns:
        raise ValueError("input csv must contain columns: movieId, info")

    # ===== Resume: 读取已有结果 =====
    existing_ids = set()
    if os.path.exists(args.output_csv):
        try:
            old_df = pd.read_csv(args.output_csv)
            if "movieId" in old_df.columns:
                existing_ids = set(old_df["movieId"].astype(str).tolist())
                print(f"[Resume] Found {len(existing_ids)} existing records, skipping them.")
        except Exception as e:
            print(f"[WARN] Failed to read existing output: {e}")

    samples = []
    missing = 0

    for _, row in df.iterrows():
        movie_id = str(row["movieId"])

        if movie_id in existing_ids:
            continue

        title = extract_title_from_info(row["info"])
        img_path = os.path.join(args.poster_dir, f"{movie_id}.jpg")

        if os.path.exists(img_path):
            samples.append((movie_id, title, img_path))
        else:
            missing += 1

    print(f"Total rows in csv: {len(df)}")
    print(f"Remaining samples to process: {len(samples)}")
    print(f"Missing posters: {missing}")

    model, processor = load_model_and_processor(
        args.model_path,
        args.dtype,
        args.max_pixels
    )

    total = len(samples)
    n_batch = math.ceil(total / args.batch_size)

    pbar = tqdm(range(n_batch), desc="Generating")

    for bi in pbar:
        batch = samples[bi * args.batch_size: (bi + 1) * args.batch_size]

        movie_ids = [x[0] for x in batch]
        titles = [x[1] for x in batch]
        img_paths = [x[2] for x in batch]

        t0 = time.time()

        try:
            descriptions = generate_batch(
                model, processor, img_paths, titles, args.max_new_tokens
            )
        except RuntimeError as e:
            print(f"\n[WARN] batch {bi} failed: {e}")
            torch.cuda.empty_cache()
            descriptions = []
            for _, title, ip in batch:
                d = generate_batch(model, processor, [ip], [title])[0]
                descriptions.append(d)

        batch_time = time.time() - t0

        # ===== 实时写入 =====
        batch_records = [
            {"movieId": mid, "description": desc}
            for mid, desc in zip(movie_ids, descriptions)
        ]

        batch_df = pd.DataFrame(batch_records)

        write_header = not os.path.exists(args.output_csv)

        batch_df.to_csv(
            args.output_csv,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8"
        )

        pbar.set_postfix({
            "bs": len(batch),
            "sec": f"{batch_time:.2f}"
        })

        if (bi + 1) % args.log_every == 0:
            print(f"\n[INFO] batch {bi+1}/{n_batch} done")
            print(f"[INFO] sample: {movie_ids[0]} -> {descriptions[0]}")

    print(f"\nDone. Output saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, default="info_subset.csv")
    parser.add_argument("--poster_dir", type=str, default="MoviePosters")
    parser.add_argument("--output_csv", type=str, default="poster_description_subset.csv")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=100)

    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()
    main(args)