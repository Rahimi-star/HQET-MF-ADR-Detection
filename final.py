
# -------------------- QUIET MODE (must be first) --------------------
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)
for name in [
    "huggingface_hub", "huggingface_hub.utils", "huggingface_hub.utils._http", "huggingface_hub.repocard",
    "transformers", "datasets", "tensorflow", "tensorflow_datasets", "absl"
]:
    logging.getLogger(name).setLevel(logging.ERROR)

# -------------------- IMPORTS --------------------
import re
import math
import time
import random
import argparse
import zipfile
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

import tensorflow_datasets as tfds
try:
    tfds.disable_progress_bar()
except Exception:
    pass

try:
    from datasets import load_dataset as hf_load_dataset
except Exception:
    hf_load_dataset = None

try:
    import urllib.request
except Exception:
    urllib = None

# Optional: quantum
try:
    import pennylane as qml
    from pennylane.qnn import TorchLayer as QmlTorchLayer
    _HAS_QML = True
except Exception:
    _HAS_QML = False

# Optional: CRF
try:
    from torchcrf import CRF
    _HAS_CRF = True
except Exception:
    _HAS_CRF = False


def print0(*a):
    print(*a, flush=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print0("✅ Device:", DEVICE)
print0("✅ torch:", torch.__version__)
print0("🧺 TFDS available:", True)
print0("🤗 HF datasets available:", hf_load_dataset is not None)
print0("⚛️ PennyLane available:", _HAS_QML)
print0("🔗 CRF available:", _HAS_CRF)


# ----------------------------- Reproducibility -----------------------------
def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ----------------------------- Config -----------------------------
@dataclass
class TrainCFG:
    base_model: str
    max_len: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 4
    grad_accum: int = 1
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    patience: int = 2

    # methodology knobs
    use_quantum: bool = False
    q_n_qubits: int = 6
    q_n_layers: int = 2

    # multi-view filtering (no timestamp)
    use_multiview_filter: bool = True
    semantic_max_dist: int = 8      # view-1: distance constraint (defensible)
    use_post_filter: bool = True    # apply semantic post-filter at inference

    # active learning
    use_active_learning: bool = False
    al_rounds: int = 3
    init_labeled_frac: float = 0.6
    query_size: int = 200
    al_pos_focus: float = 0.25  # fraction of queries biased toward likely positives

    # loss
    use_pos_weight: bool = True
    use_focal: bool = True
    focal_gamma: float = 2.0

    # tokens
    use_markers: bool = True
    add_source_token: bool = True


# ----------------------------- Text normalize -----------------------------
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


# ----------------------------- Better hand-crafted features -----------------------------
DRUG_LEX = {
    "aspirin","ibuprofen","paracetamol","acetaminophen","amoxicillin","penicillin","vaccine",
    "insulin","sertraline","zoloft","metformin","warfarin","prednisone","omeprazole"
}
ADR_LEX  = {
    "headache","migraine","nausea","vomiting","diarrhea","dizziness","fatigue","rash","itching",
    "fever","pain","swelling","allergy","anxiety","insomnia","bleeding","hypertension"
}
NEG_WORDS = {"no","not","never","none","without","can't","cannot","won't","didn't","don't","n't","failed","denied"}
POS_WORDS = {"effective","helped","cured","improved","better","relief","resolved","stable"}
UNC_WORDS = {"may","might","could","suggests","potential","possibly","perhaps","likely","unlikely"}

def _marker_counts(text: str) -> Tuple[int,int]:
    # counts of marker tags
    return (text.count("[DRUG]") + text.count("[/DRUG]"), text.count("[ADR]") + text.count("[/ADR]"))

def semantic_min_token_dist(text: str) -> Tuple[float, float]:
    """
    View-1 (semantic distance):
    - If markers exist: compute min token-distance between any DRUG span and any ADR span.
    - Else fallback: compute min distance between any lexicon drug and lexicon adr token.
    Returns: (min_dist, has_pair_flag)
    """
    t = str(text)
    toks = simple_tokens(t)
    if not toks:
        return 999.0, 0.0

    # Prefer markers if present
    if "[DRUG]" in t and "[ADR]" in t:
        # crude but stable: treat marker blocks as anchors
        # tokenize with markers removed but keep positions by splitting raw
        raw = t.replace("[/DRUG]"," [DRUG_END] ").replace("[DRUG]"," [DRUG_START] ")
        raw = raw.replace("[/ADR]"," [ADR_END] ").replace("[ADR]"," [ADR_START] ")
        rtoks = raw.split()

        drug_pos = []
        adr_pos = []
        in_drug = False
        in_adr = False
        idx = 0
        for w in rtoks:
            if w == "[DRUG_START]":
                in_drug = True
                continue
            if w == "[DRUG_END]":
                in_drug = False
                continue
            if w == "[ADR_START]":
                in_adr = True
                continue
            if w == "[ADR_END]":
                in_adr = False
                continue
            # count only word tokens
            ww = re.sub(r"[^A-Za-z']+","",w).lower()
            if ww:
                if in_drug:
                    drug_pos.append(idx)
                if in_adr:
                    adr_pos.append(idx)
                idx += 1

        if drug_pos and adr_pos:
            md = float(min(abs(i-j) for i in drug_pos for j in adr_pos))
            return md, 1.0
        return 999.0, 0.0

    # lexicon fallback
    idx_drug = [i for i,w in enumerate(toks) if w in DRUG_LEX]
    idx_adr  = [i for i,w in enumerate(toks) if w in ADR_LEX]
    if idx_drug and idx_adr:
        md = float(min(abs(i-j) for i in idx_drug for j in idx_adr))
        return md, 1.0
    return 999.0, 0.0


def compute_hand_features(text: str) -> np.ndarray:
    t = normalize_text(text)
    toks = simple_tokens(t)
    tok_set = set(toks)

    has_drug = 1.0 if tok_set.intersection(DRUG_LEX) else 0.0
    has_adr = 1.0 if tok_set.intersection(ADR_LEX) else 0.0
    drug_count = float(sum(1 for w in toks if w in DRUG_LEX))
    adr_count = float(sum(1 for w in toks if w in ADR_LEX))
    neg_count = float(sum(1 for w in toks if w in NEG_WORDS))
    pos_count = float(sum(1 for w in toks if w in POS_WORDS))
    unc_count = float(sum(1 for w in toks if w in UNC_WORDS))

    wc = float(len(toks))
    ch_len = float(len(t))
    q_marks = float(str(text).count("?"))
    e_marks = float(str(text).count("!"))
    upper_ratio = float(sum(1 for w in str(text).split() if w.isupper() and len(w) > 1) / (len(str(text).split()) + 1e-6))
    digit_ratio = float(sum(c.isdigit() for c in str(text)) / (len(str(text)) + 1e-6))
    punc_ratio = float(sum(c in ".,;:()[]{}" for c in str(text)) / (len(str(text)) + 1e-6))

    mdrug, madr = _marker_counts(str(text))
    mdist, has_pair = semantic_min_token_dist(str(text))

    # normalized distance features
    mdist_clip = float(min(mdist, 50.0))
    mdist_small = 1.0 if mdist <= 5 else 0.0
    mdist_mid = 1.0 if mdist <= 10 else 0.0

    return np.array([
        has_drug, has_adr, drug_count, adr_count,
        neg_count, pos_count, unc_count,
        wc, ch_len, upper_ratio, digit_ratio, punc_ratio,
        float(mdrug > 0), float(madr > 0),
        has_pair, mdist_clip/50.0, mdist_small, mdist_mid,
        q_marks, e_marks
    ], dtype=np.float32)

HAND_DIM = 20


# ----------------------------- BRAT parsing & labeling (TwiMed) -----------------------------
def parse_brat_ann(ann_text: str) -> Dict[str, List[Dict]]:
    entities = {}
    relations = []
    for line in ann_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("T"):
            try:
                tid, rest, mention = line.split("\t", 2)
                parts = rest.split()
                etype = parts[0]
                start, end = int(parts[1]), int(parts[2])
                entities[tid] = {"type": etype, "start": start, "end": end, "text": mention}
            except Exception:
                continue
        elif line.startswith("R"):
            try:
                _, rest = line.split("\t", 1)
                parts = rest.split()
                rtype = parts[0]
                arg1 = parts[1].split(":")[1]
                arg2 = parts[2].split(":")[1]
                relations.append({"type": rtype, "arg1": arg1, "arg2": arg2})
            except Exception:
                continue
    return {"entities": list(entities.values()), "relations": relations}


POS_REL_KEYS = (
    "outcome-negative", "outcome_negative", "ade", "adr", "adverse", "sideeffect", "side_effect",
    "drug-adr", "drug_adr", "drugade", "drug_ade", "causes", "cause", "induces", "produces"
)
NEG_REL_KEYS = ("outcome-positive", "outcome_positive", "no_effect", "cure", "benefit")


def twimed_label_from_brat(parsed: Dict) -> int:
    """
    Priority: positive (adverse/outcome-negative) wins.
    If only benefit/outcome-positive -> 0.
    If none: fallback to entity types.
    """
    rel_types = {str(r.get("type", "")).strip().lower() for r in parsed.get("relations", [])}
    has_pos = any(any(k in t for k in POS_REL_KEYS) for t in rel_types)
    has_neg = any(any(k in t for k in NEG_REL_KEYS) for t in rel_types)

    if has_pos:
        return 1
    if has_neg:
        return 0

    ent_types = {str(e.get("type", "")).strip().lower() for e in parsed.get("entities", [])}
    if any(("adr" in t) or ("ade" in t) or ("adverse" in t) for t in ent_types):
        return 1
    return 0


def apply_markers(raw_text: str, parsed: Dict) -> str:
    ents = parsed.get("entities", [])
    if not ents:
        return raw_text

    def tag_for(etype: str) -> str:
        t = etype.lower()
        if "drug" in t or "med" in t:
            return "DRUG"
        if "adr" in t or "ade" in t or "symptom" in t or "effect" in t or "adverse" in t:
            return "ADR"
        return "ENT"

    s = raw_text
    ents_sorted = sorted(ents, key=lambda x: (x["start"], x["end"]), reverse=True)
    for e in ents_sorted:
        try:
            st, en = int(e["start"]), int(e["end"])
            if st < 0 or en > len(s) or en <= st:
                continue
            tag = tag_for(str(e.get("type", "ENT")))
            s = s[:en] + f"[/{tag}]" + s[en:]
            s = s[:st] + f"[{tag}]" + s[st:]
        except Exception:
            continue
    return s


# ----------------------------- TwiMed auto-download via ZIP (NO GIT) -----------------------------
def _download_file(url: str, out_path: str):
    if urllib is None:
        raise RuntimeError("urllib not available.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def _extract_zip(zip_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def auto_get_twimed_root_zip(twimed_root: Optional[str], data_dir: str) -> str:
    if twimed_root and os.path.isdir(twimed_root):
        print0("✅ Using TwiMed at:", twimed_root)
        return twimed_root

    target = os.path.join(data_dir, "TwiMed")
    if os.path.isdir(target) and any(
        fn.endswith(".ann") or fn.endswith(".txt")
        for _, _, files in os.walk(target)
        for fn in files
    ):
        print0("✅ Using TwiMed at:", target)
        return target

    os.makedirs(data_dir, exist_ok=True)

    urls = [
        "https://github.com/nestoralvaro/TwiMed/archive/refs/heads/main.zip",
        "https://github.com/nestoralvaro/TwiMed/archive/refs/heads/master.zip",
    ]

    with tempfile.TemporaryDirectory() as td:
        ok = False
        for url in urls:
            try:
                print0("⬇️ Downloading TwiMed ZIP (" + ("main" if "main.zip" in url else "master") + ") ...")
                zp = os.path.join(td, "twimed.zip")
                _download_file(url, zp)

                ex = os.path.join(td, "extracted")
                _extract_zip(zp, ex)

                top_dirs = [os.path.join(ex, d) for d in os.listdir(ex) if os.path.isdir(os.path.join(ex, d))]
                if not top_dirs:
                    raise RuntimeError("ZIP extracted but no folder found inside.")
                src_root = top_dirs[0]

                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(src_root, target)

                ok = True
                break
            except Exception:
                continue

    if not ok:
        raise RuntimeError(
            "Failed to auto-download TwiMed ZIP.\n"
            "Fix: check internet/VPN, or download manually and pass --twimed_root PATH."
        )

    print0("✅ Auto-downloaded TwiMed at:", target)
    return target


def _collect_files(scan_root: str) -> Tuple[List[str], List[str]]:
    ann_paths, txt_paths = [], []
    for root, _, files in os.walk(scan_root):
        for fn in files:
            p = os.path.join(root, fn)
            if fn.endswith(".ann"):
                ann_paths.append(p)
            elif fn.endswith(".txt"):
                if "tools_for_twitter" in p.replace("\\", "/").lower():
                    continue
                txt_paths.append(p)
    return ann_paths, txt_paths


def _build_txt_index(txt_paths: List[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    score: Dict[str, Tuple[int, int]] = {}
    for p in txt_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            size = os.path.getsize(p)
        except Exception:
            size = 0
        low = p.replace("\\", "/").lower()
        pubmed_bonus = 1 if ("pubmed" in low) else 0
        sc = (pubmed_bonus, size)
        if stem not in index or sc > score.get(stem, (-1, -1)):
            index[stem] = p
            score[stem] = sc
    return index


def load_twimed_pubmed_only(
    twimed_root: str,
    twimed_folder: str = "gold_conflated",
    max_samples: int = 0,
    use_markers: bool = True,
    add_source_token: bool = True,
) -> pd.DataFrame:
    scan_root = os.path.join(twimed_root, twimed_folder)
    if not os.path.isdir(scan_root):
        scan_root = twimed_root

    ann_paths, txt_paths = _collect_files(scan_root)
    if len(ann_paths) == 0:
        raise RuntimeError(f"No .ann files found under: {scan_root}")

    txt_index = _build_txt_index(txt_paths)

    rows = []
    pubmed_loaded = 0
    twitter_ignored = 0
    rel_counter: Dict[str, int] = {}

    for ann_path in ann_paths:
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        txt_path = txt_index.get(stem, None)
        if not txt_path or not os.path.isfile(txt_path):
            continue

        low = (ann_path + " " + txt_path).replace("\\", "/").lower()
        is_twitter = ("twitter" in low) or (stem.isdigit() and len(stem) >= 15)
        if is_twitter:
            twitter_ignored += 1
            continue

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_txt = f.read().strip()
        if len(raw_txt) < 10:
            continue

        with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
            ann_txt = f.read()

        parsed = parse_brat_ann(ann_txt)
        for r in parsed.get("relations", []):
            rt = str(r.get("type", "")).strip().lower()
            if rt:
                rel_counter[rt] = rel_counter.get(rt, 0) + 1

        label = twimed_label_from_brat(parsed)

        # markers BEFORE normalize (offsets match raw)
        text = raw_txt
        if use_markers:
            text = apply_markers(text, parsed)
        text = normalize_text(text)

        if add_source_token:
            text = "[SRC_PUBMED] " + text

        pubmed_loaded += 1
        rows.append({"text": text, "label": int(label), "source": "TwiMed_PubMed"})

        if max_samples and len(rows) >= max_samples:
            break

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("TwiMed loading resulted in empty DataFrame.")

    print0(f"✅ TwiMed loaded (PubMed-only): {len(df)} samples")
    print0("Positive rate:", float(df["label"].mean()))
    if rel_counter:
        top = sorted(rel_counter.items(), key=lambda x: -x[1])[:5]
        print0("Top relation types:", top)
    print0(f"Twitter ignored(per requirement)={twitter_ignored}")
    return df[["text", "label", "source"]].copy()


# ----------------------------- ADE v2 loaders (TFDS first, HF fallback) -----------------------------
def _tfds_to_df(ds) -> pd.DataFrame:
    rows = []
    for ex in tfds.as_numpy(ds):
        r = {}
        for k, v in ex.items():
            if isinstance(v, (bytes, bytearray)):
                r[k] = v.decode("utf-8", errors="ignore")
            else:
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def tfds_load_concat(tfds_name: str) -> pd.DataFrame:
    obj = tfds.load(tfds_name)
    if isinstance(obj, dict):
        parts = []
        for split, ds in obj.items():
            df = _tfds_to_df(ds)
            df["split"] = split
            parts.append(df)
        return pd.concat(parts, ignore_index=True)
    return _tfds_to_df(obj)


def hf_load_concat(dataset_name: str, config_name: str) -> pd.DataFrame:
    if hf_load_dataset is None:
        raise RuntimeError("HF datasets is not installed (pip install datasets).")
    ds = hf_load_dataset(dataset_name, config_name)
    parts = []
    for split in ds.keys():
        sdf = pd.DataFrame(ds[split])
        sdf["split"] = split
        parts.append(sdf)
    return pd.concat(parts, ignore_index=True)


def load_ade_config(config_name: str, prefer_tfds: bool = True) -> pd.DataFrame:
    # Employer snippet uses TFDS:
    tfds_name = f"huggingface:ade_corpus_v2/{config_name}"

    if prefer_tfds:
        try:
            df = tfds_load_concat(tfds_name)
            print0(f"✅ ADE loaded via TFDS: {tfds_name} | rows={len(df)}")
            return df
        except Exception as e:
            print0(f"⚠️ TFDS failed for {tfds_name} → fallback HF. Reason: {type(e).__name__}: {e}")

    df = hf_load_concat("ade_corpus_v2", config_name)
    print0(f"✅ ADE loaded via HF datasets: ade_corpus_v2/{config_name} | rows={len(df)}")
    return df


def load_ade_classification(prefer_tfds: bool = True) -> pd.DataFrame:
    df = load_ade_config("Ade_corpus_v2_classification", prefer_tfds=prefer_tfds)
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["label"] = df["label"].astype(int)
    df["source"] = "Ade_corpus_v2_classification"
    return df[["text", "label", "source"]].copy()


def load_ade_relation_examples(config_name: str, prefer_tfds: bool = True) -> List[Dict]:
    """
    Returns list of dict with:
      text (str), drug (str), target (str), indexes_drug (list spans), indexes_target (list spans)
    Uses dataset-provided indexes; no cross-sentence synthetic negatives.
    """
    df = load_ade_config(config_name, prefer_tfds=prefer_tfds)

    if config_name == "Ade_corpus_v2_drug_ade_relation":
        tgt_key = "effect"
    elif config_name == "Ade_corpus_v2_drug_dosage_relation":
        tgt_key = "dosage"
    else:
        raise ValueError("Unknown relation config")

    need = {"text", "drug", tgt_key, "indexes"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"Unexpected columns for {config_name}: {list(df.columns)}")

    examples = []
    for _, row in df.iterrows():
        text = str(row["text"])
        drug = str(row["drug"])
        tgt = str(row[tgt_key])
        idx = row["indexes"]
        # idx is dict: {"drug":[{start_char,end_char}...], "effect":[...]} in TFDS/HF
        drug_spans = idx.get("drug", []) if isinstance(idx, dict) else []
        tgt_spans = idx.get(tgt_key, []) if isinstance(idx, dict) else []
        # normalize spans as list of tuples
        def spans_to_tuples(sp):
            out = []
            try:
                for s in sp:
                    out.append((int(s["start_char"]), int(s["end_char"])))
            except Exception:
                pass
            return out
        examples.append({
            "text": normalize_text(text),
            "drug": drug,
            "target": tgt,
            "drug_spans": spans_to_tuples(drug_spans),
            "tgt_spans": spans_to_tuples(tgt_spans),
            "source": config_name
        })

    print0(f"✅ ADE relation loaded: {config_name} | examples={len(examples)}")
    return examples


# ----------------------------- Safe model loading (torch<2.6 + safetensors) -----------------------------
def _torch_version_tuple():
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", str(torch.__version__))
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def safe_load_encoder(model_id: str, fallback_model_ids: Optional[List[str]] = None) -> AutoModel:
    fallback_model_ids = fallback_model_ids or [
        "allenai/scibert_scivocab_uncased",
        "distilbert-base-uncased",
        "bert-base-uncased",
    ]
    tv = _torch_version_tuple()

    try:
        return AutoModel.from_pretrained(model_id, trust_remote_code=False, use_safetensors=True)
    except Exception:
        pass

    if tv >= (2, 6, 0):
        return AutoModel.from_pretrained(model_id, trust_remote_code=False)

    for fb in fallback_model_ids:
        try:
            return AutoModel.from_pretrained(fb, trust_remote_code=False, use_safetensors=True)
        except Exception:
            continue

    raise ValueError(
        f"Could not load '{model_id}' with safetensors on torch {torch.__version__}. "
        "Upgrade torch>=2.6 OR choose a safetensors model."
    )


def build_tokenizer_and_tokens(cfg: TrainCFG):
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    extra = []
    if cfg.add_source_token:
        extra += ["[SRC_PUBMED]", "[SRC_TWITTER]"]
    if cfg.use_markers:
        extra += ["[DRUG]", "[/DRUG]", "[ADR]", "[/ADR]", "[ENT]", "[/ENT]"]
    if extra:
        tok.add_special_tokens({"additional_special_tokens": extra})
    return tok, extra


# ----------------------------- Quantum Layer (VQC via PennyLane) -----------------------------
class QuantumFeatureLayer(nn.Module):
    """
    CLS -> linear -> angles -> VQC -> expectations -> linear -> delta
    Real quantum simulation (PennyLane default.qubit).
    """
    def __init__(self, in_dim: int, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.pre = nn.Linear(in_dim, self.n_qubits)
        self.post = nn.Linear(self.n_qubits, in_dim)

        if not _HAS_QML:
            raise RuntimeError("PennyLane is not available but QuantumFeatureLayer was constructed.")

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 2)}
        self.qlayer = QmlTorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x: [B, H]
        angles = self.pre(x)
        angles = torch.tanh(angles) * math.pi
        q = self.qlayer(angles)          # [B, n_qubits]
        delta = self.post(q)             # [B, H]
        return delta


class QuantumEnhancedEncoder(nn.Module):
    def __init__(self, cfg: TrainCFG, tokenizer, extra_tokens: List[str], fallback_model_ids: Optional[List[str]]):
        super().__init__()
        self.encoder = safe_load_encoder(cfg.base_model, fallback_model_ids=fallback_model_ids)

        if extra_tokens:
            added = tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})
            if added > 0:
                self.encoder.resize_token_embeddings(len(tokenizer))

        self.drop = nn.Dropout(cfg.dropout)
        self.use_quantum = bool(cfg.use_quantum and _HAS_QML)
        self.hidden = int(self.encoder.config.hidden_size)

        if self.use_quantum:
            self.q = QuantumFeatureLayer(self.hidden, n_qubits=cfg.q_n_qubits, n_layers=cfg.q_n_layers)
            self.q_gate = nn.Sequential(nn.Linear(self.hidden * 2, self.hidden), nn.Tanh())
        else:
            self.q = None

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state                      # [B, L, H]
        cls = self.drop(hs[:, 0, :])                    # [B, H]

        if self.use_quantum and self.q is not None:
            delta = self.q(cls)                         # [B, H]
            # learned fusion (defensible, not ad-hoc constant)
            fused = self.q_gate(torch.cat([cls, delta], dim=-1))
            cls = cls + fused
            # broadcast a tiny amount of quantum context to tokens
            hs = hs + fused.unsqueeze(1) * 0.1

        return hs, cls


# ----------------------------- Multi-view sentence head -----------------------------
class MultiViewSentenceHead(nn.Module):
    def __init__(self, hidden: int, cfg: TrainCFG):
        super().__init__()
        self.use_multiview = cfg.use_multiview_filter
        self.view_mlp = nn.Sequential(
            nn.Linear(HAND_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden + (hidden if self.use_multiview else 0), hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, cls, view_feats):
        if self.use_multiview:
            v = self.view_mlp(view_feats)
            x = torch.cat([cls, v], dim=-1)
        else:
            x = cls
        return self.fuse(x).squeeze(-1)


class SentenceClassifier(nn.Module):
    def __init__(self, cfg: TrainCFG, tokenizer, extra_tokens: List[str],
                 fallback_model_ids: Optional[List[str]] = None):
        super().__init__()
        self.cfg = cfg
        self.enc = QuantumEnhancedEncoder(cfg, tokenizer, extra_tokens, fallback_model_ids=fallback_model_ids)
        self.head = MultiViewSentenceHead(self.enc.hidden, cfg)

    def forward(self, input_ids, attention_mask, view_feats):
        hs, cls = self.enc(input_ids, attention_mask)
        logits = self.head(cls, view_feats)
        return logits


# ----------------------------- Token tagging model (optional CRF) -----------------------------
TAG2ID = {"O":0, "B-DRUG":1, "I-DRUG":2, "B-TGT":3, "I-TGT":4}
ID2TAG = {v:k for k,v in TAG2ID.items()}
N_TAGS = len(TAG2ID)

class TokenTagger(nn.Module):
    def __init__(self, cfg: TrainCFG, tokenizer, extra_tokens: List[str],
                 fallback_model_ids: Optional[List[str]] = None, use_crf: bool = False):
        super().__init__()
        self.cfg = cfg
        self.enc = QuantumEnhancedEncoder(cfg, tokenizer, extra_tokens, fallback_model_ids=fallback_model_ids)
        self.proj = nn.Linear(self.enc.hidden, N_TAGS)
        self.use_crf = bool(use_crf and _HAS_CRF)
        self.crf = CRF(N_TAGS, batch_first=True) if self.use_crf else None

    def forward(self, input_ids, attention_mask, labels=None):
        hs, _ = self.enc(input_ids, attention_mask)   # [B,L,H]
        emissions = self.proj(hs)                     # [B,L,C]
        if self.use_crf and self.crf is not None:
            if labels is not None:
                loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction="mean")
                return emissions, loss
            else:
                preds = self.crf.decode(emissions, mask=attention_mask.bool())
                return preds, None
        else:
            if labels is not None:
                loss = F.cross_entropy(emissions.view(-1, N_TAGS), labels.view(-1), ignore_index=-100)
                return emissions, loss
            else:
                preds = torch.argmax(emissions, dim=-1)
                return preds, None


# ----------------------------- Loss (sentence) -----------------------------
class FocalBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets.float() + (1 - p) * (1 - targets.float())
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


def make_sentence_criterion(y_train: np.ndarray, cfg: TrainCFG):
    pw = None
    if cfg.use_pos_weight:
        n_pos = max(1.0, float((y_train == 1).sum()))
        n_neg = max(1.0, float((y_train == 0).sum()))
        pw = torch.tensor([n_neg / n_pos], device=DEVICE)
    if cfg.use_focal:
        return FocalBCEWithLogits(pos_weight=pw, gamma=cfg.focal_gamma)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


# ----------------------------- Datasets -----------------------------
class SentenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.views = np.stack([compute_hand_features(t) for t in self.df["text"].tolist()]).astype(np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["text"])
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "view_feats": torch.from_numpy(self.views[idx]).float(),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "raw_text": text,
        }


def _char_spans_to_token_labels(tokenizer, text: str, drug_spans: List[Tuple[int,int]], tgt_spans: List[Tuple[int,int]], max_len: int):
    """
    Build BIO labels using offset_mapping.
    tokens outside max_len -> ignored.
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offsets = enc["offset_mapping"].squeeze(0).tolist()
    labels = [-100] * len(offsets)

    def mark_spans(spans, b_id, i_id):
        for (st, en) in spans:
            for i, (a, b) in enumerate(offsets):
                # special tokens have (0,0)
                if a == b == 0:
                    continue
                # token overlaps span
                if not (b <= st or a >= en):
                    # first overlapping token becomes B-*
                    if labels[i] == -100 or labels[i] == TAG2ID["O"]:
                        labels[i] = b_id
                    else:
                        # if already something else, keep it
                        pass

        # second pass: convert inside to I-*
        for (st, en) in spans:
            inside = []
            for i, (a, b) in enumerate(offsets):
                if a == b == 0:
                    continue
                if not (b <= st or a >= en):
                    inside.append(i)
            if inside:
                inside = sorted(set(inside))
                labels[inside[0]] = b_id
                for j in inside[1:]:
                    if labels[j] in [-100, TAG2ID["O"]]:
                        labels[j] = i_id

    # default O for real tokens
    for i,(a,b) in enumerate(offsets):
        if a == b == 0:
            continue
        labels[i] = TAG2ID["O"]

    mark_spans(drug_spans, TAG2ID["B-DRUG"], TAG2ID["I-DRUG"])
    mark_spans(tgt_spans, TAG2ID["B-TGT"], TAG2ID["I-TGT"])

    # remove offset mapping from return (we only need labels)
    enc.pop("offset_mapping")
    return enc, torch.tensor(labels, dtype=torch.long)


class TokenDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_len: int):
        self.ex = examples
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self): return len(self.ex)

    def __getitem__(self, idx: int):
        e = self.ex[idx]
        text = str(e["text"])
        enc, labels = _char_spans_to_token_labels(
            self.tokenizer, text, e["drug_spans"], e["tgt_spans"], self.max_len
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ----------------------------- Evaluation helpers -----------------------------
def _safe_auc(y_true, probs):
    try: return float(roc_auc_score(y_true, probs))
    except Exception: return float("nan")

def _safe_auprc(y_true, probs):
    try: return float(average_precision_score(y_true, probs))
    except Exception: return float("nan")

def compute_sentence_metrics(y_true, probs, thr: float = 0.5) -> Dict[str, float]:
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "auc": _safe_auc(y_true, probs),
        "auprc": _safe_auprc(y_true, probs),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "thr": float(thr),
    }


@torch.no_grad()
def predict_sentence_probs(model: SentenceClassifier, loader: DataLoader, cfg: TrainCFG):
    model.eval()
    probs, ys, texts = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        view_feats = batch["view_feats"].to(DEVICE)
        y = batch["labels"].cpu().numpy()
        logits = model(input_ids, attn, view_feats)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(y)
        texts += batch["raw_text"]

    probs = np.concatenate(probs)
    ys = np.concatenate(ys)

    # semantic post-filter (multi-view filtering) to reduce false positives
    if cfg.use_post_filter and cfg.use_multiview_filter:
        adj = probs.copy()
        for i, tx in enumerate(texts):
            md, has_pair = semantic_min_token_dist(tx)
            # only apply when we actually detected a drug/adr pair signal
            if has_pair > 0.5 and md > float(cfg.semantic_max_dist):
                # suppress positives if semantic constraint violated
                adj[i] = adj[i] * 0.25
        probs = adj

    return probs, ys


def _entity_spans_from_tags(tag_ids: List[int]) -> List[Tuple[int,int,int]]:
    """
    returns spans: (start, end, type_id) in token index space
    type_id: 1 for DRUG, 2 for TGT
    """
    spans = []
    i = 0
    while i < len(tag_ids):
        t = tag_ids[i]
        if t == TAG2ID["B-DRUG"]:
            j = i + 1
            while j < len(tag_ids) and tag_ids[j] == TAG2ID["I-DRUG"]:
                j += 1
            spans.append((i, j, 1))
            i = j
        elif t == TAG2ID["B-TGT"]:
            j = i + 1
            while j < len(tag_ids) and tag_ids[j] == TAG2ID["I-TGT"]:
                j += 1
            spans.append((i, j, 2))
            i = j
        else:
            i += 1
    return spans


def token_entity_f1(y_true_tags: List[List[int]], y_pred_tags: List[List[int]], attn_masks: List[List[int]]) -> Dict[str,float]:
    """
    span-level micro F1 over DRUG+TGT.
    """
    tp = fp = fn = 0
    for yt, yp, am in zip(y_true_tags, y_pred_tags, attn_masks):
        L = int(sum(am))
        yt = yt[:L]
        yp = yp[:L]
        gt = set(_entity_spans_from_tags(yt))
        pr = set(_entity_spans_from_tags(yp))
        tp += len(gt & pr)
        fp += len(pr - gt)
        fn += len(gt - pr)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    return {"ent_precision": float(prec), "ent_recall": float(rec), "ent_f1": float(f1)}


# ----------------------------- Train loops (sentence) -----------------------------
def train_sentence_model(model: SentenceClassifier, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainCFG):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * math.ceil(len(train_loader) / max(1, cfg.grad_accum))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.fp16 and DEVICE == "cuda"))

    y_train = np.array(train_loader.dataset.df["label"].values, dtype=int)
    crit = make_sentence_criterion(y_train, cfg)

    best_state = None
    best_val_f1 = -1.0
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        optim.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            view_feats = batch["view_feats"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=bool(cfg.fp16 and DEVICE == "cuda")):
                logits = model(input_ids, attn, view_feats)
                loss = crit(logits, y) / max(1, cfg.grad_accum)

            scaler.scale(loss).backward()
            losses.append(float(loss.item()) * max(1, cfg.grad_accum))

            if (i + 1) % max(1, cfg.grad_accum) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

        # val eval with fixed threshold (no leakage)
        val_probs, val_y = predict_sentence_probs(model, val_loader, cfg)
        val_m = compute_sentence_metrics(val_y, val_probs, thr=0.5)

        print0(f"  epoch {ep}/{cfg.epochs} | loss={np.mean(losses):.4f} | val_f1={val_m['f1']:.4f} | time={time.time()-t0:.1f}s")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model


# ----------------------------- Train loops (token) -----------------------------
def train_token_model(model: TokenTagger, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainCFG):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.fp16 and DEVICE == "cuda"))

    best_state = None
    best_f1 = -1.0
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg.fp16 and DEVICE == "cuda")):
                _, loss = model(input_ids, attn, labels=labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(float(loss.item()))

        # val span-f1
        val_f1 = eval_token_model(model, val_loader)
        print0(f"  epoch {ep}/{cfg.epochs} | loss={np.mean(losses):.4f} | val_ent_f1={val_f1:.4f} | time={time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model


@torch.no_grad()
def eval_token_model(model: TokenTagger, loader: DataLoader) -> float:
    model.eval()
    y_true, y_pred, masks = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].cpu().numpy().tolist()

        preds, _ = model(input_ids, attn, labels=None)
        if isinstance(preds, list):  # CRF decode returns list[list[int]]
            pred_list = preds
        else:
            pred_list = preds.detach().cpu().numpy().tolist()

        mask_list = attn.detach().cpu().numpy().tolist()
        # replace ignored with O for evaluation
        cleaned_true = []
        for lab, m in zip(labels, mask_list):
            L = int(sum(m))
            arr = []
            for i in range(L):
                v = lab[i]
                arr.append(TAG2ID["O"] if v == -100 else int(v))
            cleaned_true.append(arr)
        cleaned_pred = [p[:int(sum(m))] for p, m in zip(pred_list, mask_list)]

        y_true += cleaned_true
        y_pred += cleaned_pred
        masks += mask_list

    mets = token_entity_f1(y_true, y_pred, masks)
    return float(mets["ent_f1"])


# ----------------------------- Active Learning (sentence) -----------------------------
def active_learning_loop(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: TrainCFG,
                         fallback_model_ids: List[str], seed: int):
    """
    Proper AL: split fixed before AL. AL only inside train_df.
    """
    rng = np.random.RandomState(seed)
    n = len(train_df)
    idx = np.arange(n)
    rng.shuffle(idx)

    init_n = int(max(2, cfg.init_labeled_frac * n))
    labeled_idx = set(idx[:init_n].tolist())
    pool_idx = set(idx[init_n:].tolist())

    tokenizer, extra_tokens = build_tokenizer_and_tokens(cfg)

    def build_and_train(labeled_df):
        model = SentenceClassifier(cfg, tokenizer, extra_tokens, fallback_model_ids=fallback_model_ids).to(DEVICE)
        tr_ds = SentenceDataset(labeled_df, tokenizer, cfg.max_len)
        va_ds = SentenceDataset(val_df, tokenizer, cfg.max_len)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)
        model = train_sentence_model(model, tr_ld, va_ld, cfg)
        return model, tokenizer

    # AL rounds
    for r in range(cfg.al_rounds):
        labeled_df = train_df.iloc[sorted(labeled_idx)].reset_index(drop=True)
        model, tok = build_and_train(labeled_df)

        if not pool_idx:
            break

        pool_df = train_df.iloc[sorted(pool_idx)].reset_index(drop=True)
        pool_ds = SentenceDataset(pool_df, tok, cfg.max_len)
        pool_ld = DataLoader(pool_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)

        probs, _ = predict_sentence_probs(model, pool_ld, cfg)
        # uncertainty
        uncert = np.abs(probs - 0.5)
        order_unc = np.argsort(uncert)  # most uncertain first

        q = min(cfg.query_size, len(pool_df))
        q_pos = int(q * cfg.al_pos_focus)
        q_unc = q - q_pos

        # positive-focused: among pool, take those with higher probs (but still not extremely certain)
        order_pos = np.argsort(-probs)

        chosen = set()
        for i in order_pos[:min(q_pos, len(order_pos))]:
            chosen.add(i)
        for i in order_unc:
            if len(chosen) >= q:
                break
            chosen.add(i)

        # map chosen (pool_df idx) back to original indices
        pool_list = sorted(pool_idx)
        newly = [pool_list[i] for i in chosen]
        for j in newly:
            labeled_idx.add(j)
            pool_idx.discard(j)

        print0(f"  [AL] round {r+1}/{cfg.al_rounds} | labeled={len(labeled_idx)} | pool={len(pool_idx)}")

    final_labeled_df = train_df.iloc[sorted(labeled_idx)].reset_index(drop=True)
    model, tok = build_and_train(final_labeled_df)
    return model, tok


# ----------------------------- HPO (random search, lightweight) -----------------------------
def sample_cfg(base: TrainCFG, rng: np.random.RandomState) -> TrainCFG:
    lr_choices = [1e-5, 2e-5, 3e-5, 5e-5]
    dr_choices = [0.05, 0.1, 0.2]
    wd_choices = [0.0, 0.01, 0.05]
    bs_choices = [8, 16, 32]
    ml_choices = [192, 256, 320]

    c = TrainCFG(**base.__dict__)
    c.lr = float(rng.choice(lr_choices))
    c.dropout = float(rng.choice(dr_choices))
    c.weight_decay = float(rng.choice(wd_choices))
    c.batch_size = int(rng.choice(bs_choices))
    c.max_len = int(rng.choice(ml_choices))
    # keep epochs/patience modest for HPO, but train remains stable
    return c


def hpo_select_cfg(train_df: pd.DataFrame, val_df: pd.DataFrame, base_cfg: TrainCFG,
                   fallback_model_ids: List[str], trials: int, seed: int) -> TrainCFG:
    rng = np.random.RandomState(seed)
    best_cfg = base_cfg
    best_f1 = -1.0

    for t in range(trials):
        cfg = sample_cfg(base_cfg, rng)
        tokenizer, extra_tokens = build_tokenizer_and_tokens(cfg)
        model = SentenceClassifier(cfg, tokenizer, extra_tokens, fallback_model_ids=fallback_model_ids).to(DEVICE)

        tr_ds = SentenceDataset(train_df, tokenizer, cfg.max_len)
        va_ds = SentenceDataset(val_df, tokenizer, cfg.max_len)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)

        model = train_sentence_model(model, tr_ld, va_ld, cfg)
        val_probs, val_y = predict_sentence_probs(model, va_ld, cfg)
        mets = compute_sentence_metrics(val_y, val_probs, thr=0.5)
        f1v = float(mets["f1"])
        print0(f"  [HPO] trial {t+1}/{trials} | f1={f1v:.4f} | lr={cfg.lr} bs={cfg.batch_size} len={cfg.max_len} drop={cfg.dropout}")

        if f1v > best_f1:
            best_f1 = f1v
            best_cfg = cfg

    print0(f"✅ [HPO] best f1={best_f1:.4f} | lr={best_cfg.lr} bs={best_cfg.batch_size} len={best_cfg.max_len} drop={best_cfg.dropout}")
    return best_cfg


# ----------------------------- Cross-Validation runners -----------------------------
def run_kfold_sentence(df: pd.DataFrame, name: str, base_cfg: TrainCFG, n_splits: int, seed: int,
                       fallback_model_ids: List[str], hpo_trials: int = 0) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).map(normalize_text)

    print0(f"\n================== 🚀 {name}: {n_splits}-Fold CV (Sentence) ==================")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(df, df["label"]), start=1):
        fold_seed = seed + fold
        set_seed(fold_seed, deterministic=False)

        trainval = df.iloc[trainval_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        strat = trainval["label"] if trainval["label"].nunique() > 1 else None
        train, val = train_test_split(trainval, test_size=0.1, stratify=strat, random_state=fold_seed)

        cfg = base_cfg
        if hpo_trials and hpo_trials > 0:
            cfg = hpo_select_cfg(train, val, base_cfg, fallback_model_ids=fallback_model_ids, trials=hpo_trials, seed=fold_seed)

        tokenizer, extra_tokens = build_tokenizer_and_tokens(cfg)

        # Active Learning (optional, correct split)
        if cfg.use_active_learning and cfg.al_rounds > 0:
            model, tok = active_learning_loop(train, val, cfg, fallback_model_ids=fallback_model_ids, seed=fold_seed)
            tokenizer = tok
        else:
            model = SentenceClassifier(cfg, tokenizer, extra_tokens, fallback_model_ids=fallback_model_ids).to(DEVICE)
            tr_ds = SentenceDataset(train, tokenizer, cfg.max_len)
            va_ds = SentenceDataset(val, tokenizer, cfg.max_len)
            tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            va_ld = DataLoader(va_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)
            model = train_sentence_model(model, tr_ld, va_ld, cfg)

        te_ds = SentenceDataset(test, tokenizer, cfg.max_len)
        te_ld = DataLoader(te_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)
        probs, y = predict_sentence_probs(model, te_ld, cfg)
        mets = compute_sentence_metrics(y, probs, thr=0.5)
        mets["fold"] = fold
        rows.append(mets)
        print0(f"Fold {fold} | test_f1={mets['f1']:.4f} | acc={mets['acc']:.4f}")

    return pd.DataFrame(rows)


def run_kfold_token(examples: List[Dict], name: str, base_cfg: TrainCFG, n_splits: int, seed: int,
                    fallback_model_ids: List[str], use_crf: bool = False) -> pd.DataFrame:
    print0(f"\n================== 🚀 {name}: {n_splits}-Fold CV (Token/Span) ==================")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idxs = np.arange(len(examples))
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(idxs), start=1):
        fold_seed = seed + fold
        set_seed(fold_seed, deterministic=False)

        train_ex = [examples[i] for i in tr_idx]
        test_ex = [examples[i] for i in te_idx]

        # small val split inside train
        rng = np.random.RandomState(fold_seed)
        perm = rng.permutation(len(train_ex))
        v_n = max(2, int(0.1 * len(train_ex)))
        val_ex = [train_ex[i] for i in perm[:v_n]]
        tr_ex  = [train_ex[i] for i in perm[v_n:]]

        tokenizer, extra_tokens = build_tokenizer_and_tokens(base_cfg)
        model = TokenTagger(base_cfg, tokenizer, extra_tokens, fallback_model_ids=fallback_model_ids, use_crf=use_crf).to(DEVICE)

        tr_ds = TokenDataset(tr_ex, tokenizer, base_cfg.max_len)
        va_ds = TokenDataset(val_ex, tokenizer, base_cfg.max_len)
        te_ds = TokenDataset(test_ex, tokenizer, base_cfg.max_len)

        tr_ld = DataLoader(tr_ds, batch_size=base_cfg.batch_size, shuffle=True, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=base_cfg.batch_size * 2, shuffle=False, num_workers=0)
        te_ld = DataLoader(te_ds, batch_size=base_cfg.batch_size * 2, shuffle=False, num_workers=0)

        model = train_token_model(model, tr_ld, va_ld, base_cfg)
        te_f1 = eval_token_model(model, te_ld)

        rows.append({"fold": fold, "ent_f1": te_f1})
        print0(f"Fold {fold} | test_ent_f1={te_f1:.4f}")

    return pd.DataFrame(rows)


# ----------------------------- Relation dataset: defensible negatives (no cross-sentence shuffle) -----------------------------
def make_relation_pair_classification_df(examples: List[Dict], source_name: str,
                                        max_neg_per_pos: int = 2, seed: int = 42) -> pd.DataFrame:
    """
    Defensible approach (fixes employer complaint):
    - Positive pairs come from provided (drug, target) in the sentence.
    - Negative pairs are ONLY constructed from within the SAME sentence by pairing the drug with
      other target spans inside that same text IF they exist. However ADE v2 rows only provide
      one target string; so we create negatives using "string occurrences not aligned to spans"
      is unreliable.
    => Therefore we DO NOT create cross-sentence negatives, and we DO NOT shuffle targets.
    => We instead build a 'pair-consistency' task using HARD NEGATIVES from SAME sentence:
       - negative = (drug span, random window text chunk) is not great.
    So default behavior: DO NOT build sentence-level binary from RE configs.
    We keep RE configs as token/span task via run_kfold_token (more correct & defendable).
    """
    raise RuntimeError(
        "By design: we do NOT convert ADE relation configs into binary with artificial negatives.\n"
        "Use token/span evaluation instead (run_kfold_token)."
    )


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=5)

    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--twimed_root", type=str, default=None)
    ap.add_argument("--twimed_folder", type=str, default="gold_conflated")
    ap.add_argument("--max_twimed_samples", type=int, default=0, help="0 = load all PubMed samples")

    ap.add_argument("--prefer_tfds_for_ade", type=int, default=1,
                    help="1=try tfds.load('huggingface:...') first (employer request); 0=HF datasets")

    ap.add_argument("--twimed_model", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    ap.add_argument("--ade_model", type=str, default="distilbert-base-uncased")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--fp16", type=int, default=1)

    # methodology switches
    ap.add_argument("--use_quantum", type=int, default=1, help="1=enable if PennyLane available")
    ap.add_argument("--q_n_qubits", type=int, default=6)
    ap.add_argument("--q_n_layers", type=int, default=2)

    ap.add_argument("--use_active_learning", type=int, default=1)
    ap.add_argument("--al_rounds", type=int, default=3)
    ap.add_argument("--init_labeled_frac", type=float, default=0.6)
    ap.add_argument("--query_size", type=int, default=200)
    ap.add_argument("--al_pos_focus", type=float, default=0.25)

    ap.add_argument("--use_post_filter", type=int, default=1)
    ap.add_argument("--semantic_max_dist", type=int, default=8)

    ap.add_argument("--hpo_trials", type=int, default=3, help="0 disables random-search HPO; small number recommended")

    ap.add_argument("--use_crf", type=int, default=1, help="for token/span task if torchcrf installed")

    args = ap.parse_args()
    set_seed(args.seed, deterministic=False)

    prefer_tfds = bool(args.prefer_tfds_for_ade)

    # safetensors-friendly fallback models
    twimed_fallback = ["allenai/scibert_scivocab_uncased", "distilbert-base-uncased", "bert-base-uncased"]
    ade_fallback = ["distilbert-base-uncased", "bert-base-uncased"]

    # --- Load datasets ---
    tw_root = auto_get_twimed_root_zip(args.twimed_root, data_dir=args.data_dir)
    twimed_df = load_twimed_pubmed_only(
        twimed_root=tw_root,
        twimed_folder=args.twimed_folder,
        max_samples=int(args.max_twimed_samples),
        use_markers=True,
        add_source_token=True
    )

    ade_cls = load_ade_classification(prefer_tfds=prefer_tfds)

    ade_re_ae = load_ade_relation_examples("Ade_corpus_v2_drug_ade_relation", prefer_tfds=prefer_tfds)
    ade_re_dos = load_ade_relation_examples("Ade_corpus_v2_drug_dosage_relation", prefer_tfds=prefer_tfds)

    print0("\n📦 Dataset sizes:")
    print0("  TwiMed_PubMed:", len(twimed_df))
    print0("  Ade_corpus_v2_classification:", len(ade_cls))
    print0("  Ade_corpus_v2_drug_ade_relation (RE):", len(ade_re_ae))
    print0("  Ade_corpus_v2_drug_dosage_relation (RE):", len(ade_re_dos))

    # --- Build configs ---
    base_twimed = TrainCFG(
        base_model=args.twimed_model,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fp16=bool(args.fp16),
        dropout=0.1,
        use_quantum=bool(args.use_quantum and _HAS_QML),
        q_n_qubits=int(args.q_n_qubits),
        q_n_layers=int(args.q_n_layers),
        use_multiview_filter=True,
        use_post_filter=bool(args.use_post_filter),
        semantic_max_dist=int(args.semantic_max_dist),
        use_active_learning=bool(args.use_active_learning),
        al_rounds=int(args.al_rounds),
        init_labeled_frac=float(args.init_labeled_frac),
        query_size=int(args.query_size),
        al_pos_focus=float(args.al_pos_focus),
        use_markers=True,
        add_source_token=True,
        patience=2
    )

    base_ade_sent = TrainCFG(
        base_model=args.ade_model,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=max(2, args.epochs - 1),
        fp16=bool(args.fp16),
        dropout=0.1,
        use_quantum=bool(args.use_quantum and _HAS_QML),
        q_n_qubits=int(args.q_n_qubits),
        q_n_layers=int(args.q_n_layers),
        use_multiview_filter=True,
        use_post_filter=bool(args.use_post_filter),
        semantic_max_dist=int(args.semantic_max_dist),
        use_active_learning=bool(args.use_active_learning),
        al_rounds=int(args.al_rounds),
        init_labeled_frac=float(args.init_labeled_frac),
        query_size=int(args.query_size),
        al_pos_focus=float(args.al_pos_focus),
        use_markers=False,
        add_source_token=False,
        patience=2
    )

    # For token/span tasks, markers not needed; keep same base_ade_sent but use_markers False
    base_ade_token = TrainCFG(**base_ade_sent.__dict__)
    base_ade_token.use_markers = False
    base_ade_token.add_source_token = False

    # --- Run experiments ---
    summaries = {}

    tw_res = run_kfold_sentence(
        twimed_df, "TwiMed(PubMed-only) Sentence", base_twimed,
        n_splits=args.n_splits, seed=args.seed,
        fallback_model_ids=twimed_fallback,
        hpo_trials=int(args.hpo_trials)
    )
    summaries["TwiMed(PubMed-only) Sentence"] = tw_res.describe().loc[["mean", "std"]]

    ade_res = run_kfold_sentence(
        ade_cls, "ADE v2 Classification Sentence", base_ade_sent,
        n_splits=args.n_splits, seed=args.seed + 1,
        fallback_model_ids=ade_fallback,
        hpo_trials=int(args.hpo_trials)
    )
    summaries["ADE v2 Classification Sentence"] = ade_res.describe().loc[["mean", "std"]]

    # Token/span evaluation (defensible use of RE configs; no cross-sentence synthetic negatives)
    re1 = run_kfold_token(
        ade_re_ae, "ADE v2 DRUG-AE Relation (Token/Span)", base_ade_token,
        n_splits=args.n_splits, seed=args.seed + 2,
        fallback_model_ids=ade_fallback,
        use_crf=bool(args.use_crf)
    )
    summaries["ADE v2 DRUG-AE Relation (Token/Span)"] = re1.describe().loc[["mean", "std"]]

    re2 = run_kfold_token(
        ade_re_dos, "ADE v2 DRUG-DOSE Relation (Token/Span)", base_ade_token,
        n_splits=args.n_splits, seed=args.seed + 3,
        fallback_model_ids=ade_fallback,
        use_crf=bool(args.use_crf)
    )
    summaries["ADE v2 DRUG-DOSE Relation (Token/Span)"] = re2.describe().loc[["mean", "std"]]

    print0("\n" + "=" * 90)
    print0("📊 FINAL SUMMARY (Mean ± Std over folds)")
    print0("=" * 90)
    for k, v in summaries.items():
        print0(f"\n--- {k} ---\n")
        print0(v.to_string())


if __name__ == "__main__":
    main()
