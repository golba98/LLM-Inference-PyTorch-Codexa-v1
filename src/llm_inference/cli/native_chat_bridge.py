"""Serve persistent native Codexa chat over a line-delimited JSON bridge."""

from __future__ import annotations
from llm_inference.cli.paths import asset_path, generated_path

import argparse
import json
from pathlib import Path
import sys



from llm_inference.native_chat import NativeChatEngine
from llm_inference.memory_adapter import create_memory


def build_parser() -> argparse.ArgumentParser:
    """Build the native bridge command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument(
        "--system",
        help="Optional system message. Omit it to match user-first SFT records.",
    )
    parser.add_argument("--memory", choices=("off", "ephemeral", "persistent"), default="off")
    parser.add_argument("--memory-path", type=Path, default=generated_path('data/memory/codexa.sqlite3'))
    parser.add_argument("--memory-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--memory-threshold", type=float, default=0.68)
    parser.add_argument("--user-id", default="local")
    return parser


def emit(payload: dict[str, object]) -> None:
    """Write one protocol response immediately."""

    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> None:
    """Load one model and process native chat requests until stdin closes."""

    arguments = build_parser().parse_args()
    engine = NativeChatEngine(
        checkpoint=arguments.checkpoint,
        tokenizer_path=arguments.tokenizer,
        device=arguments.device,
        maximum_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        seed=arguments.seed, do_sample=not arguments.greedy,
        repetition_penalty=arguments.repetition_penalty,
        no_repeat_ngram_size=arguments.no_repeat_ngram_size,
        system_prompt=arguments.system,
        memory=create_memory(user_id=arguments.user_id, mode=arguments.memory,
                                  path=arguments.memory_path, device=arguments.memory_device,
                                  threshold=arguments.memory_threshold),
    )
    checkpoint_name = arguments.checkpoint.parent.name
    emit({
        "type": "ready",
        "model": f"{checkpoint_name}-native",
        "device": str(engine.device),
        "context_length": engine.model.config.context_length,
    })

    for line in sys.stdin:
        request: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object.")
            request_id = str(request.get("id", ""))
            request_type = request.get("type", "chat")
            if request_type == "reset":
                engine.reset()
                emit({"type": "reset", "id": request_id, "conversation_id": engine.memory.conversation_id})
                continue
            if request_type == "memory":
                result = engine.memory_command(request.get("operation"), request.get("value"))
                emit({"type": "memory", "id": request_id, **result})
                continue
            if request_type != "chat":
                raise ValueError(f"Unsupported request type: {request_type!r}.")
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("Chat request requires a string prompt.")
            response, generated = engine.reply(prompt)
            emit({
                "type": "response",
                "id": request_id,
                "text": response,
                "finish_reason": generated.finish_reason,
                "termination_cause": generated.termination_cause,
                "generated_tokens": generated.generated_token_count,
                "memory": engine.memory.status,
            })
        except Exception as error:
            emit({
                "type": "error",
                "id": str(request.get("id", "")) if isinstance(request, dict) else "",
                "message": str(error),
            })

    engine.close()


if __name__ == "__main__":
    main()
