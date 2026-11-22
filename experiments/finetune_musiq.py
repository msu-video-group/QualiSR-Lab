"""Fine-tune the last MUSIQ feed-forward sublayer, final norm, and MOS head."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.data import FR_METRICS, NR_COLUMNS, load_experiment_data, sha256_file
from experiments.fold_features import FoldTransformer
from experiments.presets import COMPACT_STAT_COLUMNS
from experiments.run_grouped import METADATA, feature_columns, grouped_splits, make_model, write_json
from experiments.transfer import DIMS, SET120, check_partition
from qualisr.regressors import safe_corr

LR = 1e-5
MAX_EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 8
HUBER_BETA = 0.1
SCORE_SCALE = 100
PYIQA = "0.1.15.post2"
FOLD = ["split_seed", "fold"]
MODELS = ["musiq_direct", "musiq_finetuned", "qualisr", "qalign_direct"]
FEATURES = {
    "nr": NR_COLUMNS,
    "rlfn_fr": [f"{metric}_rlfn" for metric in FR_METRICS],
    "wasd": COMPACT_STAT_COLUMNS,
}
SPEC = {"scalars": [c for columns in FEATURES.values() for c in columns], "representations": ["sr"]}


def partitions(raw, cfg, pilot):
    # Keep the previously defined Set120 assignments, with other datasets
    # removed from fit rows only after generating the existing joint split.
    pool = raw.index[raw.role.eq("train")].to_numpy()
    external = raw[raw.dataset.ne(SET120)]
    if pilot:
        # Two complete sources per external dataset exercise MOS and BT evaluation.
        selected = external[["dataset", "source_group"]].drop_duplicates().groupby("dataset").head(2)
        external = external[external.source_group.isin(selected.source_group)]
    for seed in cfg["split_seeds"][:1] if pilot else cfg["split_seeds"]:
        for fold, (train, valid) in enumerate(grouped_splits(raw, pool, cfg["outer_folds"], seed), 1):
            train = train[raw.loc[train, "dataset"].eq(SET120)]
            valid = valid[raw.loc[valid, "dataset"].eq(SET120)]
            evaluate = np.concatenate([valid, external.index.to_numpy()])
            check_partition(raw, train, evaluate)
            inner = list(grouped_splits(raw, train, cfg["inner_folds"], seed))
            yield seed, fold, train, evaluate, inner
            if pilot:
                break


def partition_table(raw, seed, fold, train, evaluate, inner):
    roles = [("train", train), ("evaluate", evaluate)]
    for k, (it, iv) in enumerate(inner, 1):
        roles.extend([(f"inner_{k}_train", it), (f"inner_{k}_validate", iv)])
    return pd.concat([raw.loc[ids, ["sample_id"]].assign(split_seed=seed, fold=fold, role=role)
                      for role, ids in roles], ignore_index=True)


def digest_model(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def extract(raw, checkpoint, device, torch):
    import pyiqa

    from qualisr.features import image_to_tensor, load_image_rgb

    net = pyiqa.create_metric("musiq", device=device, pretrained_model_path=str(checkpoint)).net
    net.eval().requires_grad_(False)
    encoder = net.transformer_encoder
    last = encoder.transformer[f"encoderblock_{len(encoder.transformer) - 1}"]
    # After self-attention the rest of the block is token-wise. Cache only
    # the CLS residual entering norm2, then recompute the trainable CLS path.
    template = torch.nn.ModuleDict({
        "norm2": deepcopy(last.norm2), "mlp": deepcopy(last.mlp),
        "encoder_norm": deepcopy(encoder.encoder_norm), "head": deepcopy(net.head),
    }).requires_grad_(True)
    captured = []
    hook = last.norm2.register_forward_pre_hook(
        lambda module, args: captured.append(args[0][:, 0].clone())
    )
    direct, cls = {}, {}
    try:
        with torch.no_grad():
            for count, (idx, sample) in enumerate(raw.iterrows(), 1):
                image = image_to_tensor(load_image_rgb(Path(sample.sr_path)), device)
                original = net(image).item()
                if len(captured) != 1:
                    raise RuntimeError(f"Expected one MUSIQ CLS tensor for {sample.sample_id}")
                vector = captured.pop()
                reconstructed = predict(template, vector).item()
                if not np.isclose(original, reconstructed, atol=1e-4, rtol=1e-6):
                    raise ValueError(f"MUSIQ reconstruction differs for {sample.sample_id}")
                direct[idx] = original
                cls[idx] = vector.cpu()
                if count % 100 == 0 or count == len(raw):
                    print(f"MUSIQ frozen extraction {count}/{len(raw)}", flush=True)
    finally:
        hook.remove()
    return template, cls, pd.Series(direct)


def predict(tail, cls):
    residual = cls + tail["mlp"](tail["norm2"](cls))
    return tail["head"](tail["encoder_norm"](residual)).flatten()


def tensor_rows(cls, ids, device, torch):
    return torch.cat([cls[i] for i in ids]).to(device)


def fit_epochs(template, x, y, seed, torch):
    tail = deepcopy(template)
    optimizer = torch.optim.AdamW(tail.parameters(), lr=LR, weight_decay=0.01)
    shuffle = torch.Generator().manual_seed(seed)
    for _ in range(MAX_EPOCHS):
        tail.train()
        for batch in torch.randperm(len(y), generator=shuffle).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(predict(tail, x[batch]) / SCORE_SCALE, y[batch],
                                                       beta=HUBER_BETA)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite MUSIQ training loss")
            loss.backward()
            optimizer.step()
        yield tail.eval()


def select_epoch(template, raw, inner, cls, device, seed, torch):
    validation = []
    fits = []
    for train, valid in inner:
        x = tensor_rows(cls, train, device, torch)
        y = torch.tensor(raw.loc[train, "score"].to_numpy(), dtype=torch.float32, device=device)
        fits.append(fit_epochs(template, x, y, seed, torch))
        validation.append((tensor_rows(cls, valid, device, torch), raw.loc[valid, "score"]))
    history, best_epoch, best_score = [], 0, -np.inf
    for epoch, tails in enumerate(zip(*fits, strict=True), 1):
        with torch.no_grad():
            for k, (tail, (x, target)) in enumerate(zip(tails, validation, strict=True), 1):
                plcc, srcc = safe_corr(target, predict(tail, x).cpu().numpy())
                history.append({"epoch": epoch, "inner_fold": k, "plcc": plcc, "srcc": srcc})
        score = float(np.mean([row["srcc"] for row in history[-len(inner):]]))
        if not np.isfinite(score):
            raise ValueError("Nonfinite inner MUSIQ SRCC")
        if score > best_score:
            best_epoch, best_score = epoch, score
        if epoch - best_epoch >= PATIENCE:
            break
    return best_epoch, pd.DataFrame(history)


def feature_quantiles(raw, transformer, train, evaluate):
    quantiles = []
    families = {**FEATURES, **{family: [f"{family}_sr_{i:03d}" for i in range(n)]
                               for family, n in DIMS.items()}}
    for scope, ids in [("train", train), ("evaluate", evaluate)]:
        part = raw.loc[ids]
        features = transformer.project(part)
        for dataset, rows in features.groupby(part.dataset):
            stats = rows.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
            stats.columns = ["p05", "p25", "p50", "p75", "p95"]
            stats["finite"] = rows.count()
            stats["rows"] = len(rows)
            stats["dataset"] = dataset
            stats["scope"] = "heldout" if scope == "evaluate" and dataset == SET120 else scope
            for family, columns in families.items():
                quantiles.append(stats.loc[columns].assign(family=family).rename_axis("feature").reset_index())
    return pd.concat(quantiles, ignore_index=True)


def run(args):
    import torch

    if version("pyiqa") != PYIQA:
        raise ValueError(f"The MUSIQ split-forward implementation requires PyIQA {PYIQA}")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available GPU before using CUDA")
    device = torch.device(args.device)
    checkpoint_sha = sha256_file(args.checkpoint)
    if not checkpoint_sha.startswith("e95806b9"):
        raise ValueError("Expected the official PyIQA MUSIQ KonIQ-10k checkpoint")
    cfg = json.loads(args.config.read_text())
    settings = cfg["evaluation"]
    seed = settings["model_seed"]
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    raw, blocks, _, provenance = load_experiment_data(cfg, args.config.resolve().parent, include_all_rr=True)
    cols = METADATA + ["content_sha256", "content_kind"]
    original = pd.read_csv(args.reference_samples, float_precision="round_trip")
    pd.testing.assert_frame_equal(raw[cols], original[cols], check_exact=True)
    folds = list(partitions(raw, settings, args.pilot))
    active = np.unique(np.concatenate([np.concatenate([train, evaluate]) for _, _, train, evaluate, _ in folds]))
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    raw[cols + ["role", "declared_scale"]].to_csv(out / "samples.csv", index=False)
    write_json(out / "data_provenance.json", provenance)
    write_json(out / "run.json", {
        "config": cfg, "pilot": args.pilot, "device": str(device), "pyiqa": version("pyiqa"),
        "torch": version("torch"), "torchvision": version("torchvision"),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
        "pretraining": "ImageNet initialization; KonIQ-10k MOS",
        "trainable": ["last_transformer_block.norm2", "last_transformer_block.mlp", "encoder_norm", "head"],
        "feature_columns": feature_columns(SPEC, DIMS), "rf": {"backend": "randomforest", "params": {}},
        "epochs": MAX_EPOCHS, "patience": PATIENCE, "criterion": "mean inner MOS SRCC; earliest tie",
        "lr": LR, "weight_decay": 0.01, "batch_size": BATCH_SIZE, "huber_beta": HUBER_BETA,
        "score_scale": SCORE_SCALE, "preprocessing": "PyIQA default native + resized 224 and 384",
    })
    template, cls, direct = extract(raw.loc[active], args.checkpoint.resolve(), device, torch)
    base_hash = digest_model(template)
    assignments, searches, predictions, diagnostics, weights = [], [], [], [], []
    for split_seed, fold, train, evaluate, inner in folds:
        key = dict(split_seed=split_seed, fold=fold)
        assignments.append(partition_table(raw, split_seed, fold, train, evaluate, inner))
        chosen, history = select_epoch(template, raw, inner, cls, device, seed, torch)
        searches.append(history.assign(**key))
        x = tensor_rows(cls, train, device, torch)
        y = torch.tensor(raw.loc[train, "score"].to_numpy(), dtype=torch.float32, device=device)
        tail = next(islice(fit_epochs(template, x, y, seed, torch), chosen - 1, None))
        trained_hash = digest_model(tail)
        max_mlp_update = max((w - original_w).detach().abs().max().item()
                             for w, original_w in zip(tail["mlp"].parameters(),
                                                      template["mlp"].parameters(), strict=True))
        if trained_hash == base_hash or max_mlp_update <= 0:
            raise ValueError("MUSIQ MLP weights did not update")
        path = out / f"musiq_{split_seed}_{fold}.pt"
        torch.save({name: value.cpu() for name, value in tail.state_dict().items()}, path)
        weights.append({**key, "selected_epoch": chosen, "initial_state": base_hash,
                        "trained_state": trained_hash, "max_mlp_update": max_mlp_update,
                        "checkpoint_sha256": sha256_file(path)})
        with torch.no_grad():
            tuned = predict(tail, tensor_rows(cls, evaluate, device, torch)).cpu().numpy()
        transformer = FoldTransformer(SPEC["scalars"], blocks, DIMS, seed=seed)
        columns = feature_columns(SPEC, DIMS)
        train_x = transformer.fit_transform(raw.loc[train])
        model = make_model("randomforest", {}, seed, settings["threads"]).fit(train_x[columns], raw.loc[train, "score"])
        eval_x = transformer.transform(raw.loc[evaluate])
        qualisr = model.predict(eval_x[columns])
        predictions.append(raw.loc[evaluate, ["sample_id"]].assign(
            musiq_direct=direct.loc[evaluate], musiq_finetuned=tuned, qualisr=qualisr,
            qalign_direct=raw.loc[evaluate, "qalign"], **key
        ))
        diagnostics.append(feature_quantiles(raw, transformer, train, evaluate).assign(**key))
        print(f"seed={split_seed} fold={fold} selected_epoch={chosen}; MLP changed by {max_mlp_update:.3g}", flush=True)
    for name, parts in [("partitions", assignments), ("inner_scores", searches),
                        ("predictions", predictions), ("feature_quantiles", diagnostics)]:
        pd.concat(parts, ignore_index=True).to_csv(out / f"{name}.csv.gz", index=False)
    pd.DataFrame(weights).to_csv(out / "weights.csv", index=False)
    write_json(out / "status.json", {"status": "complete", "folds": len(folds), "image_rows": len(active)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
