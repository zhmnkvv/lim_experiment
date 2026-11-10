"""Тестовые тексты и токенайзеры для эксперимента 4."""
import os

import tiktoken
from tokenizers import Tokenizer

# Одна и та же фраза на разных языках и в разных записях.
PARALLEL = {
    "en": "Large language models process text as tokens, not as words.",
    "ru": "Большие языковые модели обрабатывают текст по токенам, а не по словам.",
    "ru_latin": "Bolshie yazykovye modeli obrabatyvayut tekst po tokenam, a ne po slovam.",
    "uk": "Великі мовні моделі обробляють текст за токенами, а не за словами.",
    "kk": "Үлкен тілдік модельдер мәтінді сөздер бойынша емес, токендер бойынша өңдейді.",
    "kk_latin": "Ülken tıldık modelder mätındı sözder boiynşa emes, tokender boiynşa öñdeidı.",
    "de": "Große Sprachmodelle verarbeiten Text als Tokens, nicht als Wörter.",
    "zh": "大型语言模型按词元而不是按单词处理文本。",
}

# Длинный абзац: проверка, держится ли соотношение ru/en на большем тексте.
PARAGRAPH = {
    "en_paragraph": (
        "Tokenization is the first step in how a language model reads text. "
        "The tokenizer splits the input into pieces from a fixed vocabulary, "
        "and the model only ever sees the numbers of those pieces. "
        "If a language is poorly represented in the vocabulary, the same idea takes more tokens, "
        "which makes requests slower and more expensive."
    ),
    "ru_paragraph": (
        "Токенизация — это первый шаг, с которого языковая модель начинает читать текст. "
        "Токенайзер разбивает входные данные на части из фиксированного словаря, "
        "и модель видит только номера этих частей. "
        "Если язык плохо представлен в словаре, та же мысль занимает больше токенов, "
        "а запросы становятся медленнее и дороже."
    ),
}

# Особые случаи, которые часто встречаются в реальных запросах.
SPECIAL = {
    "numbers": "Встреча 15.03.2026 в 14:30, бюджет 1 250 000 ₸, скидка 12,5%.",
    "code": "def add(a: int, b: int) -> int:\n    return a + b  # сумма двух чисел\n",
    "emoji": "Привет 👋 Как дела? 😊🎉🇰🇿",
    "mixed": "Запусти docker-compose up и проверь логи в Kibana через localhost:5601.",
    "typos": "Првиет, как дила? Щас буду, ток кофе вазьму.",
}

CASES = {**PARALLEL, **PARAGRAPH, **SPECIAL}


def _tiktoken(name):
    enc = tiktoken.get_encoding(name)
    return enc.encode, enc.decode


def _hf(repo):
    tok = Tokenizer.from_pretrained(repo)
    return (
        lambda text: tok.encode(text, add_special_tokens=False).ids,
        lambda ids: tok.decode(ids, skip_special_tokens=False),
    )


# name -> (фабрика, источник, поколение)
OFFLINE_TOKENIZERS = {
    "GPT-4 (cl100k_base)": (lambda: _tiktoken("cl100k_base"), "tiktoken cl100k_base", "baseline"),
    "GPT-4o (o200k_base)": (lambda: _tiktoken("o200k_base"), "tiktoken o200k_base", "baseline"),
    "Qwen2.5": (lambda: _hf("Qwen/Qwen2.5-7B-Instruct"), "Qwen/Qwen2.5-7B-Instruct", "baseline"),
    "Claude (legacy, Claude 1/2)": (lambda: _hf("Xenova/claude-tokenizer"), "Xenova/claude-tokenizer", "legacy"),
    "gpt-oss (o200k_harmony)": (lambda: _tiktoken("o200k_harmony"), "tiktoken o200k_harmony", "fresh"),
    "Qwen3.5": (lambda: _hf("Qwen/Qwen3.5-9B"), "Qwen/Qwen3.5-9B", "fresh"),
    "DeepSeek-V3.1": (lambda: _hf("deepseek-ai/DeepSeek-V3.1"), "deepseek-ai/DeepSeek-V3.1", "fresh"),
    "Llama 4 Scout": (lambda: _hf("unsloth/Llama-4-Scout-17B-16E-Instruct"), "unsloth/Llama-4-Scout-17B-16E-Instruct (зеркало)", "fresh"),
    "Gemma 3": (lambda: _hf("unsloth/gemma-3-27b-it"), "unsloth/gemma-3-27b-it (зеркало)", "fresh"),
    "Mistral Small 3.1": (lambda: _hf("mistralai/Mistral-Small-3.1-24B-Instruct-2503"), "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "fresh"),
    "GLM-4.5": (lambda: _hf("zai-org/GLM-4.5"), "zai-org/GLM-4.5", "fresh"),
    # Новейшие версии линеек на ноябрь 2026 года.
    "Qwen3.8": (lambda: _hf("Qwen/Qwen3.8-27B"), "Qwen/Qwen3.8-27B", "newest"),
    "DeepSeek-V4.1": (lambda: _hf("deepseek-ai/DeepSeek-V4.1-Flash"), "deepseek-ai/DeepSeek-V4.1-Flash", "newest"),
    "GLM-5.3": (lambda: _hf("zai-org/GLM-5.3"), "zai-org/GLM-5.3", "newest"),
    "Mistral Large 4": (lambda: _hf("mistralai/Mistral-Large-4-1T-A52B"), "mistralai/Mistral-Large-4-1T-A52B", "newest"),
    "Gemma 4": (lambda: _hf("google/gemma-4-31B-it"), "google/gemma-4-31B-it", "newest"),
}

# Токенайзеры актуальных моделей Claude не опубликованы: считаем через API count_tokens.
CLAUDE_MODELS = ["claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]


def anthropic_client():
    """Клиент Anthropic или None, если учётных данных нет."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    import anthropic

    return anthropic.Anthropic()


def claude_counter(client, model):
    """Число токенов текста без служебной обёртки сообщения.

    count_tokens считает всё сообщение целиком, поэтому вычитаем обёртку,
    измеренную на тексте из одного символа (считаем, что он занимает 1 токен).
    """

    def raw(text):
        return client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        ).input_tokens

    overhead = raw("a") - 1
    return lambda text: raw(text) - overhead, overhead
