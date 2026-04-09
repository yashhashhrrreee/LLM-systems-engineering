"""
qlora_trainer.py
================
Standalone module for fine-tuning a causal LM (GPT-2 or Llama-3-8B) to output
strictly valid JSON objects from clinical text input.

Constraint: LoRA trainable parameters must fall between 0.30% and 0.60% of total
model parameters — simulating an edge-device memory budget.

Usage:
    python src/qlora_trainer.py --model gpt2 --epochs 2 --output ./adapter_weights
"""

import json
import re
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100
MAX_LENGTH = 256
DEFAULT_MODEL = "gpt2"

# NCBI Disease-schema samples (mirrors load_dataset("ncbi_disease") structure)
# tokens: list[str], ner_tags: list[int]  (0=O, 1=B-Disease, 2=I-Disease)
TRAINING_SAMPLES = [
    (
        ["Identification", "of", "APC2", ",", "a", "homologue", "of", "the",
         "adenomatous", "polyposis", "coli", "tumour", "suppressor", "."],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 0],
    ),
    (
        ["Germline", "mutations", "in", "BRCA1", "predispose", "carriers", "to",
         "breast", "cancer", "and", "ovarian", "cancer", "."],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0],
    ),
    (
        ["Loss", "of", "NF1", "function", "leads", "to", "neurofibromatosis",
         "type", "1", "and", "juvenile", "myelomonocytic", "leukemia", "."],
        [0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 1, 2, 2, 0],
    ),
    (
        ["TP53", "mutations", "are", "associated", "with", "Li-Fraumeni",
         "syndrome", "and", "colorectal", "carcinoma", "."],
        [0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0],
    ),
    (
        ["PTEN", "deletions", "cause", "Cowden", "syndrome", "and", "increase",
         "risk", "of", "endometrial", "carcinoma", "."],
        [0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2, 0],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_disease_string(tokens: list[str], ner_tags: list[int]) -> str:
    """Return a space-joined string of all disease-tagged tokens."""
    return " ".join(tok for tok, tag in zip(tokens, ner_tags) if tag in (1, 2))


def build_prompt(tokens: list[str], ner_tags: list[int]) -> str:
    """
    Format a single sample into the strict clinical-parser prompt.
    Loss is computed ONLY on the ### OUTPUT section.
    """
    raw_text = " ".join(tokens)
    disease = extract_disease_string(tokens, ner_tags)
    subject = tokens[0] if tokens else "Unknown"
    outcome = "disease identified" if disease else "no disease identified"
    output_json = json.dumps({"subject": subject, "disease": disease, "outcome": outcome})

    return (
        "### SYSTEM: You are a clinical data parser. Extract the core entities "
        "from the medical text below. You must output ONLY a valid JSON object "
        'with the keys "subject", "disease", and "outcome".\n\n'
        f"### INPUT: {raw_text}\n\n"
        f"### OUTPUT: {output_json}"
    )


def print_trainable_parameters(model) -> float:
    """Print and return the trainable parameter percentage."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    assert 0.30 <= pct <= 0.60, (
        f"Parameter budget violated: {pct:.4f}% is outside [0.30%, 0.60%]"
    )
    return pct


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClinicalJSONDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    has_gpu = torch.cuda.is_available()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ) if has_gpu else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if has_gpu else None,
    )

    # LoRA rank r=16 on c_attn → ~0.47% trainable for GPT-2 (Conv1D layers)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tokenization with loss masking
# ---------------------------------------------------------------------------

def tokenize_and_mask(text: str, tokenizer) -> dict[str, torch.Tensor]:
    """Mask all input-prompt tokens; loss computed only on JSON output."""
    split_marker = "### OUTPUT: "
    split_idx = text.index(split_marker)
    input_part = text[: split_idx + len(split_marker)]
    output_part = text[split_idx + len(split_marker):]

    input_ids_prompt = tokenizer.encode(input_part, add_special_tokens=False)
    input_ids_output = tokenizer.encode(output_part, add_special_tokens=False)
    full_ids = input_ids_prompt + input_ids_output

    labels = [IGNORE_INDEX] * len(input_ids_prompt) + input_ids_output

    pad_len = MAX_LENGTH - len(full_ids)
    input_ids = full_ids[:MAX_LENGTH] + [tokenizer.pad_token_id] * max(0, pad_len)
    attention_mask = [1] * min(len(full_ids), MAX_LENGTH) + [0] * max(0, pad_len)
    labels = labels[:MAX_LENGTH] + [IGNORE_INDEX] * max(0, pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, tokenizer, epochs: int = 2, lr: float = 3e-4, batch_size: int = 2):
    device = next(model.parameters()).device

    tokenized = [
        tokenize_and_mask(build_prompt(tokens, tags), tokenizer)
        for tokens, tags in TRAINING_SAMPLES
    ]
    loader = DataLoader(
        ClinicalJSONDataset(tokenized),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} — loss: {total_loss / len(loader):.4f}")

    return model


# ---------------------------------------------------------------------------
# Inference & validation
# ---------------------------------------------------------------------------

def generate_and_validate(model, tokenizer, text: str) -> dict:
    """
    Run inference on unseen clinical text, extract JSON from output,
    and validate via json.loads(). Raises ValueError on parse failure.
    """
    prompt = (
        "### SYSTEM: You are a clinical data parser. Extract the core entities "
        "from the medical text below. You must output ONLY a valid JSON object "
        'with the keys "subject", "disease", and "outcome".\n\n'
        f"### INPUT: {text}\n\n"
        "### OUTPUT: "
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_section = generated.split("### OUTPUT:")[-1].strip()

    json_match = re.search(r"\{.*?\}", output_section, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in model output:\n{output_section}")

    parsed = json.loads(json_match.group(0))
    print("✅ JSON parsed successfully!")
    print("Parsed output:", parsed)
    return parsed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA Clinical JSON Fine-tuner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("\nTraining...")
    model = train(model, tokenizer, epochs=args.epochs, lr=args.lr)

    print("\nMerging adapter weights...")
    fused = model.merge_and_unload()

    unseen_text = (
        "A 45-year-old male presented with acute myocardial infarction following "
        "prolonged chest pain. Troponin levels were critically elevated. "
        "Emergency PCI was performed with successful revascularization."
    )
    print("\nValidating on unseen text...")
    generate_and_validate(fused, tokenizer, unseen_text)


if __name__ == "__main__":
    main()
