from adapter import LinearWithAdapter
from utiles import *

def main():
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, get_linear_schedule_with_warmup)
    from sklearn.metrics import accuracy_score, f1_score

    MODEL = "roberta-base"
    MAX_LEN, BS, EPOCHS, LR = 256, 16, 3, 1e-3   # adapters like a high LR
    BOTTLENECK = 64
    TRAIN_N, EVAL_N = 4000, 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    ds = load_dataset("stanfordnlp/imdb")
    tok = AutoTokenizer.from_pretrained(MODEL)
    prep = lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN)
    train = ds["train"].shuffle(seed=42).select(range(TRAIN_N)).map(
        prep, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    test = ds["test"].shuffle(seed=42).select(range(EVAL_N)).map(
        prep, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    coll = DataCollatorWithPadding(tok)
    tr_dl = DataLoader(train, batch_size=BS, shuffle=True, collate_fn=coll)
    ev_dl = DataLoader(test, batch_size=BS * 2, collate_fn=coll)

    # ---- model ----
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    add_adapters(model, rank=BOTTLENECK)
    freeze_for_adapter_tuning(model)
    model.to(device)
    tr, tot = count_params(model)
    print(f"Trainable: {tr:,} / {tot:,} ({100 * tr / tot:.2f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    total_steps = len(tr_dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    @torch.no_grad()
    def evaluate():
        model.eval()
        P, L = [], []
        for b in ev_dl:
            b = {k: v.to(device) for k, v in b.items()}
            lg = model(**{k: v for k, v in b.items() if k != "labels"}).logits
            P += lg.argmax(-1).tolist()
            L += b["labels"].tolist()
        return {"acc": accuracy_score(L, P), "f1": f1_score(L, P)}

    print("Before training:", evaluate())

    # ---- train ----
    for ep in range(EPOCHS):
        model.train()
        run = 0.0
        for i, b in enumerate(tr_dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(**b).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            if i % 50 == 0:
                print(f"ep {ep+1} step {i}/{len(tr_dl)} loss {run/50:.4f}")
                run = 0.0
        print(f"== epoch {ep+1}:", evaluate())

    # ---- save only the adapter weights ----
    torch.save(adapter_state_dict(model), "adapter_weights.pt")
    print("Saved adapter_weights.pt")

    # ---- reload into a fresh model to prove it round-trips ----
    fresh = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    add_adapters(fresh, rank=BOTTLENECK)
    load_adapter_state_dict(fresh, torch.load("adapter_weights.pt"))
    fresh.to(device).eval()
    texts = ["An absolute masterpiece, I loved every minute.",
             "Boring, predictable, and far too long."]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        probs = fresh(**enc).logits.softmax(-1).cpu()
    for t, p in zip(texts, probs):
        print(f"{t!r} -> {'POSITIVE' if p.argmax() else 'NEGATIVE'} ({p.max():.3f})")


if __name__ == "__main__":
    main()