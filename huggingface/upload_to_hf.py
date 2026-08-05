"""
upload_to_hf.py  —  publish Pi-of-AI's large binaries to the Hugging Face Hub.

These are the files that DON'T belong in git (the GGUF model + the SFT dataset).
Putting them on the Hub gives them a proper home AND lets you delete the local
copies to reclaim disk.

This script performs NO uploads until YOU run it with your own token. It only
touches repos under YOUR account.

    pip install -U huggingface_hub           # already installed here
    huggingface-cli login                    # paste a token: https://hf.co/settings/tokens (WRITE scope)
    python huggingface/upload_to_hf.py --user YOUR_HF_USERNAME

    # preview what WOULD happen, upload nothing:
    python huggingface/upload_to_hf.py --user YOUR_HF_USERNAME --dry-run

    # make the repos private:
    python huggingface/upload_to_hf.py --user YOUR_HF_USERNAME --private
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# (local file  ->  path inside the HF repo)
MODEL_FILES = {
    REPO_ROOT / "rules_baker" / "web" / "smoke_housestyle.gguf": "smoke_housestyle.gguf",
    HERE / "MODEL_CARD.md": "README.md",
}
DATASET_FILES = {
    REPO_ROOT / "rules_baker" / "datasets" / "rules_sft.jsonl": "rules_sft.jsonl",
    HERE / "DATASET_CARD.md": "README.md",
}


def push(api, repo_id: str, repo_type: str, files: dict, private: bool, dry_run: bool) -> None:
    present = {src: dst for src, dst in files.items() if src.exists()}
    missing = [src.name for src in files if not src.exists()]
    if missing:
        print(f"  (skipping missing: {', '.join(missing)})")
    if not present:
        print(f"  nothing to upload for {repo_id} — skipped.\n")
        return

    print(f"  {repo_type} repo: {repo_id}  ({'PRIVATE' if private else 'public'})")
    for src, dst in present.items():
        mb = src.stat().st_size / 1e6
        print(f"    {src.name:<28} -> {dst:<20} ({mb:.1f} MB)")
    if dry_run:
        print("  [dry-run] no changes made.\n")
        return

    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    for src, dst in present.items():
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=repo_id, repo_type=repo_type)
    kind = "datasets" if repo_type == "dataset" else ""
    print(f"  done -> https://huggingface.co/{kind + '/' if kind else ''}{repo_id}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload Pi-of-AI binaries to the HF Hub.")
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--model-repo", default="pi-of-ai-rules-baker", help="model repo name")
    ap.add_argument("--dataset-repo", default="pi-of-ai-rules-sft", help="dataset repo name")
    ap.add_argument("--private", action="store_true", help="create repos as private")
    ap.add_argument("--dry-run", action="store_true", help="show plan, upload nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    if not args.dry_run:
        who = api.whoami()  # fails loudly if not logged in
        print(f"Logged in as: {who['name']}\n")

    print("== Model ==")
    push(api, f"{args.user}/{args.model_repo}", "model", MODEL_FILES, args.private, args.dry_run)
    print("== Dataset ==")
    push(api, f"{args.user}/{args.dataset_repo}", "dataset", DATASET_FILES, args.private, args.dry_run)

    if not args.dry_run:
        print("All done. You can now delete the local copies to reclaim disk if you like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
