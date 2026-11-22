"""Set120-only prediction-head adaptation of the remaining default NR metrics."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.data import load_experiment_data, require, sha256_file
from experiments.finetune_musiq import (
    BATCH_SIZE,
    HUBER_BETA,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    PYIQA,
    digest_model,
    partition_table,
    partitions,
)
from experiments.run_grouped import METADATA, write_json
from experiments.transfer import SET120
from qualisr.regressors import safe_corr

METRICS = ("arniqa", "unique", "paq2piq", "qalign")
# Convert native output units to the existing parser's normalized MOS target.
# These constants use published model scales, never external target statistics.
SCALES = {"arniqa": (0., 1.), "unique": (0., 1.), "paq2piq": (0., 100.), "qalign": (1., 4.)}
TRAINABLE = {
    "arniqa": "KonIQ regressor weights and bias; frozen dual-scale encoder",
    "unique": "mean row of pretrained bilinear quality head; frozen encoder and uncertainty row",
    "paq2piq": "pretrained quality MLP and BN affine parameters; fixed BN running statistics and dropout disabled",
    "qalign": "final language RMSNorm and five pretrained quality-token output rows; frozen visual/language blocks",
}


def init_adapter(metric, device, torch):
    import pyiqa

    # Convert Q-Align to FP32 on CPU before its single GPU transfer. This
    # avoids retaining temporary FP16 allocations on the selected GPU.
    load_device = torch.device("cpu") if metric == "qalign" else device
    net = pyiqa.create_metric(metric, device=load_device).net.eval().requires_grad_(False)
    if metric == "arniqa":
        # The released regressor is TorchScript; copy its actual parameters into
        # an ordinary Linear so hooks/autograd do not depend on JIT internals.
        head = torch.nn.Linear(net.feat_dim * 2, 1).to(device)
        with torch.no_grad():
            head.weight.copy_(net.regressor.weights)
            head.bias.copy_(net.regressor.biases)
        tail = torch.nn.ModuleDict({"head": head})
        boundary = net.encoder
    elif metric == "unique":
        head = torch.nn.Linear(net.fc.in_features, 1).to(device)
        with torch.no_grad():
            head.weight.copy_(net.fc.weight[:1])
            head.bias.copy_(net.fc.bias[:1])
        tail, boundary = torch.nn.ModuleDict({"head": head}), net.fc
    elif metric == "paq2piq":
        # Pooling/flattening are frozen; train only the existing downstream head.
        tail, boundary = torch.nn.ModuleDict({"head": deepcopy(net.head[2:])}), net.head[2]
    else:
        # FP32 matches the earlier CPU Q-Align benchmark and keeps head
        # reconstruction in the same precision on either extraction device.
        net.model.float()
        net.model_dtype = torch.float32
        net.to(device)
        net.model.weight_tensor = torch.tensor([5., 4., 3., 2., 1.], device=device,
                                               dtype=net.model_dtype)
        ids = net.model.preferential_ids_
        head = torch.nn.Linear(net.model.lm_head.in_features, len(ids), bias=False).to(device)
        with torch.no_grad():
            head.weight.copy_(net.model.lm_head.weight[ids])
        tail = torch.nn.ModuleDict({"norm": deepcopy(net.model.model.norm).float(), "head": head})
        boundary = net.model.model.norm
    return net, tail.float().eval().requires_grad_(True), boundary


def predict(tail, x, metric, torch):
    if metric == "arniqa":
        return (tail["head"](x).flatten() - 1.) / 99.
    if metric == "qalign":
        logits = tail["head"](tail["norm"](x))
        return logits.softmax(-1) @ x.new_tensor([5., 4., 3., 2., 1.])
    return tail["head"](x).flatten()


def extract_one(net, tail, boundary, image, metric, torch):
    captured = []
    hook = None
    if metric == "arniqa":
        hook = boundary.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
    else:
        def capture(module, args):
            x = args[0][:, -1] if metric == "qalign" else args[0]
            captured.append(x.detach().clone())
        hook = boundary.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            original = float(net(image).item())
            if metric == "arniqa":
                require(len(captured) == 2, "Expected native and half-scale ARNIQA features")
                x = torch.hstack([torch.nn.functional.normalize(value, dim=1) for value in captured])
                vector = x.reshape(1, net.feat_dim * 2)
            else:
                require(len(captured) == 1, f"Expected one {metric} head input")
                # Q-Align internally uses inference_mode: clone outside it so
                # cached features can subsequently participate in autograd.
                vector = captured[0].clone()
            vector = vector.float()
            reconstructed = float(predict(tail, vector, metric, torch).item())
        require(np.isfinite(original) and np.isfinite(reconstructed), f"Nonfinite {metric} score")
        require(np.isclose(original, reconstructed, atol=1e-4, rtol=1e-5),
                f"{metric} head reconstruction differs: {original} vs {reconstructed}")
        return vector.cpu(), original, abs(original - reconstructed)
    finally:
        if hook is not None:
            hook.remove()


def fit_epochs(template, x, y, metric, seed, torch):
    tail = deepcopy(template).eval()
    optimizer = torch.optim.AdamW(tail.parameters(), lr=LR, weight_decay=0.01)
    shuffle = torch.Generator().manual_seed(seed)
    offset, scale = SCALES[metric]
    for _ in range(MAX_EPOCHS):
        # Evaluation mode deliberately holds pretrained BN statistics fixed and
        # disables dropout, while gradients still update the head parameters.
        for batch in torch.randperm(len(y), generator=shuffle).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            score = (predict(tail, x[batch], metric, torch) - offset) / scale
            loss = torch.nn.functional.smooth_l1_loss(score, y[batch], beta=HUBER_BETA)
            require(bool(torch.isfinite(loss)), f"Nonfinite {metric} training loss")
            loss.backward()
            optimizer.step()
        yield tail


def select_epoch(template, raw, inner, vectors, metric, device, seed, torch):
    fits, validation = [], []
    for train, valid in inner:
        x = torch.cat([vectors[i] for i in train]).to(device)
        y = torch.tensor(raw.loc[train, "score"].to_numpy(), dtype=torch.float32, device=device)
        fits.append(fit_epochs(template, x, y, metric, seed, torch))
        validation.append((torch.cat([vectors[i] for i in valid]).to(device), raw.loc[valid, "score"]))
    history, best_epoch, best_score = [], 0, -np.inf
    for epoch, tails in enumerate(zip(*fits, strict=True), 1):
        with torch.no_grad():
            for k, (tail, (x, target)) in enumerate(zip(tails, validation, strict=True), 1):
                plcc, srcc = safe_corr(target, predict(tail, x, metric, torch).cpu().numpy())
                history.append(dict(epoch=epoch, inner_fold=k, plcc=plcc, srcc=srcc))
        score = float(np.mean([row["srcc"] for row in history[-len(inner):]]))
        require(np.isfinite(score), f"Nonfinite inner {metric} SRCC")
        if score > best_score:
            best_epoch, best_score = epoch, score
        if epoch - best_epoch >= PATIENCE:
            break
    return best_epoch, pd.DataFrame(history)


def run(args):
    import torch

    from qualisr.features import image_to_tensor, load_image_rgb

    require(version("pyiqa") == PYIQA, f"Requires PyIQA {PYIQA}")
    require(args.device != "cuda" or torch.cuda.device_count() == 1, "Expose exactly one free GPU")
    cfg = json.loads(args.config.read_text())
    parent_run = json.loads((args.parent / "run.json").read_text())
    require(cfg == parent_run["config"] and not parent_run["pilot"], "Expected matching full MUSIQ parent")
    require(json.loads((args.parent / "analysis/verification.json").read_text())["status"] == "passed",
            "MUSIQ parent must be verified")
    require(json.loads((args.parent / "status.json").read_text())["status"] == "complete", "Parent incomplete")
    # The caller may supply a portable sha256sum manifest relative to TORCH_HOME.
    if args.checkpoint_manifest.suffix == ".sha256":
        for line in args.checkpoint_manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            path = Path(torch.hub.get_dir()).parent / name
            require(sha256_file(path) == digest, f"Pretrained checkpoint differs: {path}")
    settings, device = cfg["evaluation"], torch.device(args.device)
    seed = settings["model_seed"]
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Occupy the selected GPU immediately, before the several-minute data audit
    # during which another shared-server job could otherwise claim the device.
    print(f"{args.metric}: initializing model before data audit", flush=True)
    net, template, boundary = init_adapter(args.metric, device, torch)
    initial = digest_model(template)
    raw, _, _, provenance = load_experiment_data(cfg, args.config.resolve().parent, include_all_rr=True)
    cols = METADATA + ["content_sha256", "content_kind", "role", "declared_scale"]
    pd.testing.assert_frame_equal(raw[cols], pd.read_csv(args.parent / "samples.csv", float_precision="round_trip")[cols],
                                  check_exact=True)
    folds = list(partitions(raw, settings, args.pilot))
    assignments = pd.concat([partition_table(raw, s, f, t, e, inner) for s, f, t, e, inner in folds],
                            ignore_index=True)
    # Verify exact parent outer and inner IDs, before any fitting or selection.
    prior = pd.read_csv(args.parent / "partitions.csv.gz")
    prior = prior.merge(assignments[["split_seed", "fold"]].drop_duplicates(), validate="many_to_one")
    if not args.pilot:
        order = ["split_seed", "fold", "role", "sample_id"]
        pd.testing.assert_frame_equal(assignments.sort_values(order).reset_index(drop=True),
                                      prior.sort_values(order).reset_index(drop=True))
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    raw[cols].to_csv(out / "samples.csv", index=False)
    assignments.to_csv(out / "partitions.csv.gz", index=False)
    write_json(out / "data_provenance.json", provenance)
    write_json(out / "run.json", {
        "metric": args.metric, "config": cfg, "pilot": args.pilot, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": version("torch"), "pyiqa": version("pyiqa"), "trainable": TRAINABLE[args.metric],
        "lr": LR, "epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "weight_decay": 0.01, "huber_beta": HUBER_BETA, "score_offset_scale": SCALES[args.metric],
        "criterion": "mean inner MOS SRCC; earliest tie", "model_seed": seed,
        "preprocessing": "unchanged PyIQA default; no extra crops/resizing or external normalization",
        "precision": "FP32 frozen extraction and head adaptation",
        "parent": str(args.parent), "parent_predictions_sha256": sha256_file(args.parent / "predictions.csv.gz"),
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "adaptation_scope": "prediction head only; frozen encoders; capacities differ between metrics",
    })
    vectors, frozen, errors = {}, {}, []
    set_ids = np.unique(np.concatenate([t for _, _, t, _, _ in folds] + [
        e[raw.loc[e, "dataset"].eq(SET120)] for _, _, _, e, _ in folds]))

    def extract(idx):
        image = image_to_tensor(load_image_rgb(Path(raw.loc[idx, "sr_path"])), device)
        vector, direct, error = extract_one(net, template, boundary, image, args.metric, torch)
        frozen[idx] = direct
        errors.append(error)
        return vector

    for idx in set_ids:
        vectors[idx] = extract(idx)
    print(f"{args.metric}: extracted {len(vectors)} Set120 images", flush=True)
    tails, searches, weights = [], [], []
    for split_seed, fold, train, _, inner in folds:
        key = dict(split_seed=split_seed, fold=fold)
        chosen, history = select_epoch(template, raw, inner, vectors, args.metric, device, seed, torch)
        searches.append(history.assign(**key))
        x = torch.cat([vectors[i] for i in train]).to(device)
        y = torch.tensor(raw.loc[train, "score"].to_numpy(), dtype=torch.float32, device=device)
        tail = next(islice(fit_epochs(template, x, y, args.metric, seed, torch), chosen - 1, None))
        update = max((p - q).detach().abs().max().item()
                     for p, q in zip(tail.parameters(), template.parameters(), strict=True))
        trained = digest_model(tail)
        require(trained != initial and np.isfinite(update) and update > 0, "Head parameters did not update")
        path = out / f"{args.metric}_{split_seed}_{fold}.pt"
        torch.save({n: v.detach().cpu().clone() for n, v in tail.state_dict().items()}, path)
        weights.append({**key, "selected_epoch": chosen, "initial_state": initial, "trained_state": trained,
                        "max_parameter_update": update, "checkpoint_sha256": sha256_file(path)})
        tails.append(tail)
        print(f"{args.metric}: {split_seed}/{fold} selected epoch {chosen}; parameter update {update:.6g}", flush=True)
    pd.concat(searches, ignore_index=True).to_csv(out / "inner_scores.csv.gz", index=False)
    pd.DataFrame(weights).to_csv(out / "weights.csv", index=False)
    # Stream external head inputs once through all fifteen fitted tails. Only
    # Set120 training features are cached; UNIQUE's bilinear features are large.
    targets = [set(e) for _, _, _, e, _ in folds]
    active = sorted(set.union(*targets))
    predictions = []
    for count, idx in enumerate(active, 1):
        vector = vectors[idx] if idx in vectors else extract(idx)
        with torch.no_grad():
            for (s, f, *_), tail, ids in zip(folds, tails, targets, strict=True):
                if idx in ids:
                    value = float(predict(tail, vector.to(device), args.metric, torch).item())
                    require(np.isfinite(value), "Nonfinite fine-tuned prediction")
                    predictions.append(dict(sample_id=raw.loc[idx, "sample_id"], split_seed=s, fold=f,
                                            direct=frozen[idx], finetuned=value))
        if count % 100 == 0 or count == len(active):
            print(f"{args.metric}: paired evaluation {count}/{len(active)}", flush=True)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv.gz", index=False)
    write_json(out / "status.json", {"status": "complete", "folds": len(folds),
                                     "image_rows": len(set(set_ids) | set(active)),
                                     "max_reconstruction_error": max(errors)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "parent", "checkpoint-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--pilot", action="store_true")
    with threadpool_limits(limits=2):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
