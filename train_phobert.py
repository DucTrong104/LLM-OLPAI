"""
PhoBERT Aspect-Conditioned Regression for OLP'AI 2026 - Subtask 1

Input JSONL:
{"id": "...", "text": "...", "aspects": [
  {"aspect": "...", "valence": 7.5, "arousal": 4.2}
]}

Install:
pip install torch transformers sentencepiece scikit-learn numpy

Run:
python train_phobert.py \
  --train_path train.jsonl \
  --output_dir outputs/phobert_acr \
  --epochs 6 \
  --batch_size 16 \
  --max_length 160

The script:
1. Loads and validates JSONL.
2. Preprocesses text (conservative normalization).
3. Flattens sentence-aspect pairs.
4. Splits by sentence id to prevent leakage.
5. Fine-tunes PhoBERT with a 2-dimensional regression head.
6. Selects the best checkpoint by the competition's 2D RMSE.
7. Saves model, tokenizer, metrics, and validation predictions.
"""

import argparse
import json
import os
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ---------------------------
# 1. Reproducibility
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility may reduce speed slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# 2. Text preprocessing
# ---------------------------

ABBREVIATIONS = {
    # Only unambiguous, context-independent forms.
    "đg": "đang",
    "đc": "được",
    "dc": "được",
    "mn": "mọi người",
    "ngta": "người ta",
    "cskh": "chăm sóc khách hàng",
    "ko": "không",
    "kh": "không",
    "vs": "với",
    "cx": "cũng",
}

EMOJI_MAP = {
    "😭": " emoji_cry ",
    "😡": " emoji_angry ",
    "🤦‍♂️": " emoji_facepalm ",
    "🤦‍♀️": " emoji_facepalm ",
    "👍": " emoji_positive ",
    "❤️": " emoji_heart ",
    "❤": " emoji_heart ",
    "😂": " emoji_laugh ",
    "😢": " emoji_sad ",
    "😔": " emoji_sad ",
}

EMOTICON_PATTERNS = [
    (r"(?i):\)+", " emoticon_laugh "),
    (r"(?i)=\)+", " emoticon_laugh "),
    (r":\(", " emoticon_sad "),
    (r":D", " emoticon_happy "),
    (r";\)", " emoticon_wink "),
]

SPECIAL_TOKENS = [
    "emoji_cry",
    "emoji_angry",
    "emoji_facepalm",
    "emoji_positive",
    "emoji_heart",
    "emoji_laugh",
    "emoji_sad",
    "emoticon_laugh",
    "emoticon_sad",
    "emoticon_happy",
    "emoticon_wink",
]


def normalize_text(
    text: str,
    normalize_abbreviations: bool = True,
    normalize_emoji: bool = True,
    normalize_repeated_chars: bool = False,
    lowercase: bool = True,
) -> str:
    """Conservative normalization; does not remove Vietnamese diacritics."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    # Convert selected emoji into readable special tokens.
    if normalize_emoji:
        for emoji_char, token in EMOJI_MAP.items():
            text = text.replace(emoji_char, token)

        for pattern, token in EMOTICON_PATTERNS:
            text = re.sub(pattern, token, text)

    # Replace only known abbreviations, using word boundaries.
    if normalize_abbreviations:
        for short, full in ABBREVIATIONS.items():
            text = re.sub(
                rf"(?<!\w){re.escape(short)}(?!\w)",
                full,
                text,
                flags=re.IGNORECASE,
            )

    # Optional ablation: reduce character repetitions.
    # Disabled by default because repeated characters can encode emphasis.
    if normalize_repeated_chars:
        text = re.sub(r"([^\W\d_])\1{2,}", r"\1", text, flags=re.UNICODE)

    # Compress excessive punctuation, but preserve punctuation type.
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    if lowercase:
        text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# 3. Load and validate JSONL
# ---------------------------

def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}")

            required = {"id", "text", "aspects"}
            missing = required - set(item.keys())
            if missing:
                raise ValueError(
                    f"Line {line_no} missing required fields: {missing}"
                )

            if not isinstance(item["aspects"], list):
                raise ValueError(f"'aspects' must be a list at line {line_no}")

            records.append(item)

    if not records:
        raise ValueError("No records found in training JSONL.")

    return records


def flatten_records(records, args):
    samples = []
    for item in records:
        text_original = item["text"]
        text_clean = normalize_text(
            text_original,
            normalize_abbreviations=args.normalize_abbreviations,
            normalize_emoji=args.normalize_emoji,
            normalize_repeated_chars=args.normalize_repeated_chars,
            lowercase=args.lowercase,
        )

        for aspect_item in item["aspects"]:
            if not {"aspect", "valence", "arousal"}.issubset(aspect_item):
                raise ValueError(
                    f"Invalid aspect record in sentence id={item['id']}: "
                    f"{aspect_item}"
                )

            valence = float(aspect_item["valence"])
            arousal = float(aspect_item["arousal"])

            if not (1.0 <= valence <= 9.0 and 1.0 <= arousal <= 9.0):
                raise ValueError(
                    f"Target out of [1,9] for id={item['id']}: "
                    f"{valence}, {arousal}"
                )

            aspect = str(aspect_item["aspect"])
            if not aspect.strip():
                raise ValueError(f"Empty aspect in sentence id={item['id']}")

            samples.append({
                "id": str(item["id"]),
                "text_original": text_original,
                "text": text_clean,
                "aspect": aspect,  # Preserve original aspect string.
                "valence": valence,
                "arousal": arousal,
            })

    if not samples:
        raise ValueError("No aspect-level training samples found.")

    return samples


# ---------------------------
# 4. Group split by original sentence id
# ---------------------------

def split_by_group(samples, val_size=0.15, seed=42):
    groups = np.asarray([x["id"] for x in samples])
    indices = np.arange(len(samples))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(indices, groups=groups))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    train_ids = {x["id"] for x in train_samples}
    val_ids = {x["id"] for x in val_samples}
    assert train_ids.isdisjoint(val_ids), "Group leakage detected!"

    return train_samples, val_samples


# ---------------------------
# 5. Dataset and tokenizer
# ---------------------------

class AspectRegressionDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        # Sentence-pair input: [CLS] aspect [SEP] text [SEP]
        encoded = self.tokenizer(
            item["aspect"],
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                [item["valence"], item["arousal"]],
                dtype=torch.float32,
            ),
        }

        # Some tokenizers may return token_type_ids; PhoBERT generally
        # does not require them, so the model handles them optionally.
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return result


# ---------------------------
# 6. Transfer learning model
# ---------------------------

class PhoBERTAspectRegressor(nn.Module):
    def __init__(self, model_name, vocab_size, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Needed because special tokens were added to tokenizer.
        self.encoder.resize_token_embeddings(vocab_size)

        hidden_size = self.encoder.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        raw_output = self.regression_head(cls_vector)

        # Map unconstrained output to [1,9].
        predictions = 1.0 + 8.0 * torch.sigmoid(raw_output)
        return predictions


# ---------------------------
# 7. Metrics
# ---------------------------

def calculate_metrics(y_true, y_pred):
    # Competition metric: sqrt(mean((Vpred-Vgold)^2 + (Apred-Agold)^2))
    rmse_2d = np.sqrt(
        np.mean(
            (y_pred[:, 0] - y_true[:, 0]) ** 2
            + (y_pred[:, 1] - y_true[:, 1]) ** 2
        )
    )
    rmse_valence = np.sqrt(
        np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)
    )
    rmse_arousal = np.sqrt(
        np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)
    )

    return {
        "rmse_2d": float(rmse_2d),
        "rmse_valence": float(rmse_valence),
        "rmse_arousal": float(rmse_arousal),
    }


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(**batch)
            predictions.append(preds.cpu().numpy())
            targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    return calculate_metrics(y_true, y_pred), y_true, y_pred


# ---------------------------
# 8. Train
# ---------------------------

def train(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/7] Loading JSONL...")
    records = load_jsonl(args.train_path)
    samples = flatten_records(records, args)

    print(f"Sentences: {len(records)}")
    print(f"Aspect-level samples: {len(samples)}")

    print("\n[2/7] Group train/validation split...")
    train_samples, val_samples = split_by_group(
        samples,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"Train aspect samples: {len(train_samples)}")
    print(f"Validation aspect samples: {len(val_samples)}")
    print(
        "Train sentence ids:",
        len({x['id'] for x in train_samples}),
        "| Validation sentence ids:",
        len({x['id'] for x in val_samples}),
    )

    print("\n[3/7] Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
    )

    if args.normalize_emoji:
        tokenizer.add_special_tokens({
            "additional_special_tokens": SPECIAL_TOKENS
        })

    train_dataset = AspectRegressionDataset(
        train_samples, tokenizer, args.max_length
    )
    val_dataset = AspectRegressionDataset(
        val_samples, tokenizer, args.max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("\n[4/7] Initializing transfer learning model...")
    model = PhoBERTAspectRegressor(
        args.model_name,
        len(tokenizer),
        dropout=args.dropout,
    ).to(device)

    # Full fine-tuning: encoder + regression head.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.MSELoss()
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    print("\n[5/7] Fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(**batch)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        metrics, y_true, y_pred = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_MSE={train_loss:.5f} | "
            f"val_RMSE_2D={metrics['rmse_2d']:.5f} | "
            f"val_RMSE_V={metrics['rmse_valence']:.5f} | "
            f"val_RMSE_A={metrics['rmse_arousal']:.5f}"
        )

        if metrics["rmse_2d"] < best_rmse:
            best_rmse = metrics["rmse_2d"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint_dir = output_dir / "best_model"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model.encoder.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save(
                model.regression_head.state_dict(),
                checkpoint_dir / "regression_head.pt",
            )

            with open(checkpoint_dir / "model_config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": args.model_name,
                        "max_length": args.max_length,
                        "dropout": args.dropout,
                        "best_epoch": best_epoch,
                        "best_val_rmse_2d": best_rmse,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            np.savez(
                output_dir / "best_validation_predictions.npz",
                y_true=y_true,
                y_pred=y_pred,
            )
            print("  -> Saved new best checkpoint.")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("\n[6/7] Saving training history...")
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_rmse_2d": best_rmse,
        "num_sentences": len(records),
        "num_aspect_samples": len(samples),
        "num_train_aspect_samples": len(train_samples),
        "num_val_aspect_samples": len(val_samples),
        "seed": args.seed,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "preprocessing": {
            "normalize_abbreviations": args.normalize_abbreviations,
            "normalize_emoji": args.normalize_emoji,
            "normalize_repeated_chars": args.normalize_repeated_chars,
            "lowercase": args.lowercase,
        },
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[7/7] Finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Best model directory: {output_dir / 'best_model'}")


# ---------------------------
# 9. CLI arguments
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/phobert_acr")

    parser.add_argument(
        "--model_name",
        type=str,
        default="vinai/phobert-base-v2",
    )
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=2)

    # Preprocessing flags: all can be switched off for ablation.
    parser.add_argument(
        "--normalize_abbreviations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize_emoji",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize_repeated_chars",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
