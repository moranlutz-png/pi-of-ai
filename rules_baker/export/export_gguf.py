"""
export_gguf.py  —  Merge the LoRA adapter and export to GGUF for edge serving.

Turns the trained adapter into a single quantized GGUF that Ollama / llama.cpp
can serve locally over an OpenAI-compatible API. This is the "burn it to the SD
card" step of the Raspberry-Pi-of-AI workflow.

    python export/export_gguf.py --config configs/qwen_coder_7b.yaml

Then create the Ollama model (see export/Modelfile):
    ollama create qwen-coder-housestyle -f export/Modelfile
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger("rules_baker.export")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Merge adapter + export GGUF.")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    adapter_dir = Path(cfg["train"]["output_dir"]) / "adapter"
    quant = cfg["export"]["gguf_quant"]

    from unsloth import FastLanguageModel  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=cfg["student"]["max_seq_length"],
        load_in_4bit=cfg["student"]["load_in_4bit"],
    )
    gguf_dir = Path(cfg["train"]["output_dir"]) / "gguf"
    # Unsloth merges LoRA into base and writes a quantized GGUF in one call.
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)
    logger.info("GGUF (%s) written -> %s", quant, gguf_dir)
    logger.info("Next: ollama create %s -f export/Modelfile",
                cfg["export"]["ollama_model_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
