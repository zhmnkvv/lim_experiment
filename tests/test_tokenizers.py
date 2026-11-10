"""Тесты эксперимента 4: свойства токенайзеров и согласованность сохранённых результатов."""
import json
from functools import cache

import pytest
from conftest import EXP4
from lib import CASES, CLAUDE_MODELS, OFFLINE_TOKENIZERS, PARALLEL, anthropic_client, claude_counter

TOKENIZER_NAMES = list(OFFLINE_TOKENIZERS)


@cache
def tokenizer(name):
    return OFFLINE_TOKENIZERS[name][0]()


@cache
def saved_results():
    return json.loads((EXP4 / "results.json").read_text(encoding="utf-8"))


def test_cases_are_not_empty():
    assert all(text.strip() for text in CASES.values())


def test_parallel_set_has_english_baseline():
    assert "en" in PARALLEL and len(PARALLEL) >= 5


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
@pytest.mark.parametrize("case", list(CASES))
def test_roundtrip(name, case):
    """Декодирование токенов возвращает исходный текст без потерь."""
    encode, decode = tokenizer(name)
    assert decode(encode(CASES[case])) == CASES[case]


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
def test_encoding_is_deterministic(name):
    encode, _ = tokenizer(name)
    text = CASES["ru_paragraph"]
    assert encode(text) == encode(text)


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
def test_token_count_is_positive_and_bounded_by_bytes(name):
    """Байтовые BPE-токенайзеры не дают больше токенов, чем байт в тексте."""
    encode, _ = tokenizer(name)
    for text in CASES.values():
        assert 0 < len(encode(text)) <= len(text.encode())


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
def test_saved_results_match_fresh_run(name):
    """results.json совпадает с повторным подсчётом, то есть результаты воспроизводимы."""
    encode, _ = tokenizer(name)
    saved = saved_results()["tokenizers"][name]["counts"]
    assert saved == {case: len(encode(text)) for case, text in CASES.items()}


def test_saved_results_cover_all_cases_and_tokenizers():
    data = saved_results()
    assert set(data["cases"]) == set(CASES)
    assert set(data["tokenizers"]) == set(OFFLINE_TOKENIZERS)


def test_tables_generated():
    tables = (EXP4 / "tables.md").read_text(encoding="utf-8")
    for name in TOKENIZER_NAMES:
        assert name in tables


@pytest.mark.skipif(anthropic_client() is None, reason="нет ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN")
@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_claude_count_tokens(model):
    count, overhead = claude_counter(anthropic_client(), model)
    assert overhead >= 0
    assert 0 < count(PARALLEL["en"]) < count(PARALLEL["ru"])
