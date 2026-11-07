"""Эксперимент 2: русский на латинице и казахский (кириллица и латиница) против русского и английского."""
import json
import sys
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent

PHRASES = {
    "en": "Large language models process text as tokens, not as words.",
    "ru": "Большие языковые модели обрабатывают текст по токенам, а не по словам.",
    "ru_latin": "Bolshie yazykovye modeli obrabatyvayut tekst po tokenam, a ne po slovam.",
    "kk": "Үлкен тілдік модельдер мәтінді сөздер бойынша емес, токендер бойынша өңдейді.",
    "kk_latin": "Ülken tıldık modelder mätındı sözder boiynşa emes, tokender boiynşa öñdeidı.",
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
        k: {"text": t, "chars": len(t), "words": len(t.split()), "utf8_bytes": len(t.encode())}
        for k, t in PHRASES.items()
    },
    "tokenizers": {},
}
for name, (encode, decode) in TOKENIZERS.items():
    row = {k: measure(t, encode, decode) for k, t in PHRASES.items()}
    for k in PHRASES:
        row[k]["vs_en"] = round(row[k]["tokens"] / row["en"]["tokens"], 2)
    results["tokenizers"][name] = row

(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

for k, v in results["phrases"].items():
    print(k, v["chars"], v["words"], v["utf8_bytes"])
for name, row in results["tokenizers"].items():
    print(f"\n{name}")
    for k in PHRASES:
        r = row[k]
        print(f"  {k:9} tokens={r['tokens']:3} vs_en={r['vs_en']} tok/word={r['tokens_per_word']} chars/tok={r['chars_per_token']}")
        print("     ", " | ".join(r["pieces"]))
