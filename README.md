# LLM-Inference

Native completion, chat, context composition and export.

Owns native generation, stopping, chat history, bridge protocol and retrieval-to-prompt composition. Plain inference does not require Memory or Specialist. Enable memory explicitly with the memory extra and configure its separate worker. The GPT-2/Llama export adapters are experimental: their runtime semantics do not faithfully preserve all Codexa architectures. Use native export for fidelity.

## Development

In the sibling workspace, use `../LLM-From-Scratch/run.py --repo LLM-Inference test`.
This selects the existing environment and sibling package sources without installing dependencies.
For a separately installed checkout, run `python -m pytest` after provisioning the documented dependencies and exact sibling version 0.1.0. These packages are local and not published to PyPI.

## Entry points

- `python -m llm_inference.cli.generate --help`
- `python -m llm_inference.cli.chat_native --help`
- `python -m llm_inference.cli.native_chat_bridge --help`
- `python -m llm_inference.cli.export_chat_native --help`
- `python -m llm_inference.cli.export_codexa_lmstudio --help`
- `python -m llm_inference.cli.export_codexa_llama_lmstudio --help`

## Integration and assets

`../LLM-From-Scratch/compatibility.json` records the complete tested version set.
Checkpoint weights, tokenizers, datasets and generated logs are referenced by path; none are distributed in this package. Preserve tokenizer fingerprints and architecture lineage. Source provenance is in PROVENANCE.md.

## Validation and limitations

See the central VALIDATION.md for commands, results and unverified large-model checks.
The original project is preserved unchanged. No model promotion, training pipeline or remote publishing occurs as part of extraction.
# LLM-Inference-PyTorch-Codexa-v1

## Canonical workspace integration

This repository remains independently versioned at its existing remote and is pinned as a sibling in LLM-From-Scratch/compatibility.json. Integration decisions live in ../LLM-From-Scratch/documentation/training/SESSION_DECISIONS.md. Historical assets are external inputs; never commit weights, datasets or recovery snapshots.
