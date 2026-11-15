"""Snapshot public resources for dedicated SR-IQA comparisons, without loading models.

Uses only the standard library. HTTP failures are fatal; absent search results
are evidence about the recorded queries, not proof that private code is absent.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import platform
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PFIQA = "xinyouu/PFIQA"
DATASET = "Lighting-YXLI/ISRGen-QA"
TIMM_URL = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/v0.5.4/timm/models/vision_transformer.py"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "QualiSR-Lab-resource-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def github(path: str):
    return json.loads(fetch("https://api.github.com/" + path))


def collection(path: str) -> list:
    """Read all pages of a GitHub collection, preserving explicit HTTP errors."""
    rows = []
    for page in range(1, 101):
        batch = github(f"{path}?per_page=100&page={page}")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise ValueError(f"Collection exceeds the audit page limit: {path}")


def repository(name: str) -> dict:
    info = github(f"repos/{name}")
    commit = github(f"repos/{name}/commits/{info['default_branch']}")["sha"]
    tree = github(f"repos/{name}/git/trees/{commit}?recursive=1")
    if tree["truncated"]:
        raise ValueError(f"Incomplete repository tree: {name}")
    files = [{"path": item["path"], "git_blob": item["sha"], "size": item["size"]}
             for item in tree["tree"] if item["type"] == "blob"]
    releases = collection(f"repos/{name}/releases")
    return {
        "repository": name, "commit": commit, "default_branch": info["default_branch"],
        "url": info["html_url"], "files": files,
        "weight_files": [item["path"] for item in files
                         if Path(item["path"]).suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors", ".onnx"}],
        "releases": [{"tag": item["tag_name"], "url": item["html_url"], "body": item["body"],
                      "assets": [asset["browser_download_url"] for asset in item["assets"]]}
                     for item in releases],
    }


def source_record(url: str) -> tuple[str, dict]:
    data = fetch(url)
    return data.decode("utf-8"), {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def forward_signature(source: str, class_name: str) -> inspect.Signature:
    """Inspect positional API arity without importing or running upstream code."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    if method.args.defaults or method.args.kwonlyargs or method.args.vararg or method.args.kwarg:
        raise ValueError("Review the signature inspector for this changed upstream API")
    return inspect.Signature([inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_ONLY)
                              for arg in method.args.posonlyargs + method.args.args])


def inspect_pfiqa(sources: dict[str, str], timm_source: str) -> dict:
    signature = forward_signature(timm_source, "VisionTransformer")
    calls = []
    for filename in ("train.py", "test.py"):
        for node in ast.walk(ast.parse(sources[filename])):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                    and node.func.attr == "vit"):
                try:
                    signature.bind(None, *[None for _ in node.args], **{kw.arg: None for kw in node.keywords})
                except TypeError as error:
                    calls.append({"file": filename, "line": node.lineno, "compatible": False,
                                  "signature_binding_error": str(error)})
                else:
                    calls.append({"file": filename, "line": node.lineno, "compatible": True})
    if not calls:
        raise ValueError("No ViT calls found; review changed upstream code")
    load_paths = []
    for node in ast.walk(ast.parse(sources["test.py"])):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                and node.func.attr == "load" and node.args and isinstance(node.args[0], ast.Constant)):
            load_paths.append({"line": node.lineno, "path": node.args[0].value})
    return {
        "type": "static source/API audit, not neural inference or training",
        "pinned_requirements": sources["requirements.txt"].splitlines(),
        "timm_vision_transformer_forward": str(signature),
        "vit_calls": sorted(calls, key=lambda row: (row["file"], row["line"])),
        "test_literal_checkpoint_paths": load_paths,
        "checkpoint_save_expression": [
            {"line": i, "text": line.strip()} for i, line in enumerate(sources["train.py"].splitlines(), 1)
            if "self.regressor.stat==e_dict()" in line
        ],
        "python_syntax": "all downloaded PFIQA Python files parsed successfully",
    }


def challenge_exposure(samples: Path, public_lists: dict[str, str], output: Path) -> dict:
    """Join official challenge image partitions to existing IDs, without labels."""
    membership = {}
    for partition, text in public_lists.items():
        for line in text.splitlines():
            name = Path(line.split("#")[0]).stem
            if name in membership:
                raise ValueError(f"Duplicate challenge image across public lists: {name}")
            membership[name] = partition
    with samples.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["dataset"] == "isrgenqa"]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Expected unique ISRGen-QA sample IDs in the shared manifest")
    if {row["role"] for row in rows} != {"external"}:
        raise ValueError("Expected the paper's external ISRGen-QA population")
    if {row["method"] for row in rows} != set(membership) or len(rows) != len(membership):
        raise ValueError("Shared manifest image names do not match the public challenge lists")
    matched = [{"sample_id": row["sample_id"], "public_partition": membership[row["method"]],
                "content_sha256": row["content_sha256"]} for row in rows]
    with (output / "challenge_exposure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matched[0]))
        writer.writeheader()
        writer.writerows(matched)
    source_sets = {partition: {row["content_sha256"] for row in matched if row["public_partition"] == partition}
                   for partition in public_lists}
    return {
        "manifest": str(samples.resolve()), "manifest_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "matched_images": len(matched), "images_per_partition": dict(Counter(row["public_partition"] for row in matched)),
        "source_identity": "Existing manifest HR-file content_sha256, grouping all LR scales of an HR source",
        "sources_per_partition": {key: len(value) for key, value in source_sets.items()},
        "validation_sources_also_in_train": len(source_sets["val"] & source_sets["train"]),
        "test_sources_also_in_train": len(source_sets["test"] & source_sets["train"]),
        "interpretation": "Public partition membership; not proof of any particular checkpoint's training history",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; existing paths are rejected")
    parser.add_argument("--samples", type=Path, help="Grouped run's samples.csv for challenge-exposure alignment")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    pfiqa, dataset = repository(PFIQA), repository(DATASET)
    sources, records = {}, {}
    for item in pfiqa["files"]:
        if item["path"].endswith(".py") or item["path"] in {"README.md", "requirements.txt", "LICENSE"}:
            url = f"https://raw.githubusercontent.com/{PFIQA}/{pfiqa['commit']}/{item['path']}"
            source, record = source_record(url)
            if item["path"].endswith(".py"):
                ast.parse(source, filename=item["path"])
            sources[item["path"]], records[item["path"]] = source, record
    timm_source, timm_record = source_record(TIMM_URL)
    searches = []
    for query in ("PFIQA", "VQualA SR", "ISRGC", '"BLIP-2" quality',
                  '"residual-guided" "super-resolution"'):
        url = "search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": 100})
        result = github(url)
        searches.append({"query": query, "url": "https://api.github.com/" + url,
                         "total_count": result["total_count"], "incomplete_results": result["incomplete_results"],
                         "repositories": [item["full_name"] for item in result["items"]]})
    author_repos = collection("users/ShishunTian/repos")
    forks = [repository(item["full_name"]) for item in collection(f"repos/{PFIQA}/forks")]
    hf_url = "https://huggingface.co/api/models?search=PFIQA"
    challenge_url = "https://arxiv.org/html/2509.06413v1"
    challenge, challenge_record = source_record(challenge_url)
    challenge_links = sorted(set(re.findall(r'href="(https?://[^"\s]+)"', challenge)))
    public_lists, list_records = {}, {}
    for partition in ("train", "val", "test"):
        url = f"https://raw.githubusercontent.com/{DATASET}/{dataset['commit']}/{partition}.txt"
        public_lists[partition], list_records[partition] = source_record(url)
    result = {
        "collected_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "status": "resource_audit_complete", "pfiqa": pfiqa, "pfiqa_sources": records,
        "pfiqa_source_checks": inspect_pfiqa(sources, timm_source), "pinned_timm_source": timm_record,
        "pfiqa_forks": forks, "isrgenqa": dataset, "repository_searches": searches,
        "ydy_leader_public_repositories": [{"name": item["full_name"], "description": item["description"]}
                                           for item in author_repos],
        "huggingface_pfiqa": {"url": hf_url, "models": [item["id"] for item in json.loads(fetch(hf_url))]},
        "challenge_report": {**challenge_record, "external_links": challenge_links},
        "challenge_lists": list_records,
        "limits": ["Search absence is limited to these endpoints/queries and collection time",
                   "No model weights downloaded, dependencies installed, or neural code executed",
                   "No pretraining-overlap clearance is implied by a missing checkpoint"],
    }
    if options.samples:
        result["shared_manifest_exposure"] = challenge_exposure(options.samples, public_lists, options.output)
    path = options.output / "resource_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}; {len(pfiqa['weight_files'])} PFIQA weight files, {len(pfiqa['releases'])} releases")


if __name__ == "__main__":
    main()
