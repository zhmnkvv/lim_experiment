"""Эксперимент 3: случайность. Один и тот же вопрос три раза, каждый раз в новом чате."""
import difflib
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Придумай название для кофейни"
RUNS = 3
SEEDS = [1, 2, 3]
MAX_NEW_TOKENS = 300

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
gen_cfg = model.generation_config
sampling = {k: getattr(gen_cfg, k) for k in ("temperature", "top_p", "top_k", "repetition_penalty")}


def new_chat(do_sample, seed):
    # Новый чат: история состоит только из одного сообщения пользователя.
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    torch.manual_seed(seed)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=do_sample, max_new_tokens=MAX_NEW_TOKENS,
                             **({} if do_sample else {"temperature": None, "top_p": None, "top_k": None}))
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return {
        "seed": seed,
        "answer": tok.decode(new_ids, skip_special_tokens=True).strip(),
        "tokens": len(new_ids),
        "seconds": round(time.time() - start, 1),
    }


def compare(runs):
    answers = [r["answer"] for r in runs]
    return {
        "all_identical": len(set(answers)) == 1,
        "pairwise_similarity": {
            f"{i + 1}-{j + 1}": round(difflib.SequenceMatcher(None, answers[i], answers[j]).ratio(), 2)
            for i, j in itertools.combinations(range(len(answers)), 2)
        },
    }


results = {"model": MODEL, "prompt": PROMPT, "sampling_params": sampling, "modes": {}}
for mode, do_sample in (("sampling", True), ("greedy", False)):
    runs = [new_chat(do_sample, seed) for seed in SEEDS[:RUNS]]
    results["modes"][mode] = {"runs": runs, **compare(runs)}

(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("sampling params:", sampling)
for mode, data in results["modes"].items():
    print(f"\n===== {mode}: identical={data['all_identical']} similarity={data['pairwise_similarity']}")
    for i, r in enumerate(data["runs"], 1):
        print(f"\n--- chat {i} (seed={r['seed']}, {r['tokens']} tokens, {r['seconds']}s)\n{r['answer']}")
