"""Эксперимент 1: одна и та же фраза на русском и английском через разные токенайзеры."""
import json
import sys
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent

PHRASES = {
    "ru": "Большие языковые модели обрабатывают текст по токенам, а не по словам.",
    "en": "Large language models process text as tokens, not as words.",
}


def tiktoken_backend(name):
    enc = tiktoken.get_encoding(name)
    return enc.encode, lambda ids: enc.decode(ids)


def hf_backend(repo):
    tok = Tokenizer.from_pretrained(repo)

    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    return encode, lambda ids: tok.decode(ids, skip_special_tokens=False)


TOKENIZERS = {
    "gpt2 (GPT-2)": tiktoken_backend("gpt2"),
    "cl100k_base (GPT-4)": tiktoken_backend("cl100k_base"),
    "o200k_base (GPT-4o)": tiktoken_backend("o200k_base"),
    "Qwen2.5": hf_backend("Qwen/Qwen2.5-7B-Instruct"),
}


def measure(text, encode, decode):
    ids = encode(text)
    assert decode(ids) == text, "текст не восстановился после декодирования"
    return {
        "tokens": len(ids),
        "tokens_per_word": round(len(ids) / len(text.split()), 2),
        "chars_per_token": round(len(text) / len(ids), 2),
        "ids": ids,
        "pieces": [decode([i]) for i in ids],
    }


results = {
    "phrases": {
        lang: {"text": t, "chars": len(t), "words": len(t.split()), "utf8_bytes": len(t.encode())}
        for lang, t in PHRASES.items()
    },
    "tokenizers": {},
}
for name, (encode, decode) in TOKENIZERS.items():
    row = {lang: measure(t, encode, decode) for lang, t in PHRASES.items()}
    row["ru_en_ratio"] = round(row["ru"]["tokens"] / row["en"]["tokens"], 2)
    results["tokenizers"][name] = row

(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print({k: {m: v[m] for m in ("chars", "words", "utf8_bytes")} for k, v in results["phrases"].items()})
for name, row in results["tokenizers"].items():
    print(f"\n{name}: ru={row['ru']['tokens']} en={row['en']['tokens']} ratio={row['ru_en_ratio']}")
    print(f"  ru tok/word={row['ru']['tokens_per_word']} chars/tok={row['ru']['chars_per_token']}")
    print(f"  en tok/word={row['en']['tokens_per_word']} chars/tok={row['en']['chars_per_token']}")
    print("  ru:", " | ".join(row["ru"]["pieces"]))
    print("  en:", " | ".join(row["en"]["pieces"]))
