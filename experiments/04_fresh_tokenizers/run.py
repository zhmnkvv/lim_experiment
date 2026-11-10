"""Эксперимент 4: расширенный набор тестов на свежих токенайзерах и Claude."""
import json
import sys
from pathlib import Path

from lib import CASES, CLAUDE_MODELS, OFFLINE_TOKENIZERS, PARALLEL, SPECIAL, anthropic_client, claude_counter

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent

results = {
    "cases": {k: {"text": t, "chars": len(t), "utf8_bytes": len(t.encode())} for k, t in CASES.items()},
    "tokenizers": {},
    "claude_api": {"status": None, "models": {}},
}

for name, (factory, source, generation) in OFFLINE_TOKENIZERS.items():
    encode, decode = factory()
    row = {"source": source, "generation": generation, "counts": {}, "roundtrip_ok": {}, "pieces": {}}
    for case, text in CASES.items():
        ids = encode(text)
        row["counts"][case] = len(ids)
        row["roundtrip_ok"][case] = decode(ids) == text
        row["pieces"][case] = [decode([i]) for i in ids]
    results["tokenizers"][name] = row

client = anthropic_client()
if client is None:
    results["claude_api"]["status"] = "skipped: no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN"
else:
    results["claude_api"]["status"] = "ok"
    for model in CLAUDE_MODELS:
        count, overhead = claude_counter(client, model)
        results["claude_api"]["models"][model] = {
            "message_overhead": overhead,
            "counts": {case: count(text) for case, text in CASES.items()},
        }

(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

# Сводные таблицы в Markdown из тех же данных.
all_counts = {n: r["counts"] for n, r in results["tokenizers"].items()}
all_counts.update({f"Claude API: {m}": r["counts"] for m, r in results["claude_api"]["models"].items()})


def table(cases, ratio_to=None):
    lines = ["| Токенайзер | " + " | ".join(cases) + " |", "|---|" + "---:|" * len(cases)]
    for name, counts in all_counts.items():
        if ratio_to:
            cells = [f"{counts[c] / counts[ratio_to]:.2f}" for c in cases]
        else:
            cells = [str(counts[c]) for c in cases]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


md = [
    "### Число токенов: одна фраза на разных языках", table(list(PARALLEL)),
    "### Во сколько раз больше, чем по-английски", table([c for c in PARALLEL if c != "en"], ratio_to="en"),
    "### Абзац: русский и английский", table(["en_paragraph", "ru_paragraph"]),
    "### Особые случаи", table(list(SPECIAL)),
]
(OUT / "tables.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
print("\n\n".join(md))
print("\nClaude API:", results["claude_api"]["status"])
bad = {n: [c for c, ok in r["roundtrip_ok"].items() if not ok] for n, r in results["tokenizers"].items()}
print("roundtrip failures:", {n: c for n, c in bad.items() if c})
