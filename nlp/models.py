"""Local Hugging Face models behind a small interface, so tests can swap in a fake."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from config import settings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger(__name__)

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ALL_MODELS = (SENTIMENT_MODEL, EMOTION_MODEL, ZERO_SHOT_MODEL)

EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
# Inputs are padded to a multiple of this many tokens so MPS sees few distinct shapes.
PAD_MULTIPLE = 64

Scores = dict[str, float]


class Analyzer(Protocol):
    device: str

    def sentiment(self, texts: list[str]) -> list[Scores]: ...

    def emotion(self, texts: list[str]) -> list[Scores]: ...

    def zero_shot(
        self, texts: list[str], labels: list[str], template: str, multi_label: bool
    ) -> list[Scores]: ...

    def model_hashes(self) -> dict[str, str]: ...

    def release_memory(self) -> None: ...


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference in ("cpu", "mps", "cuda"):
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HFAnalyzer:
    """Runs the three models directly (tokenizer + model), loading each once on first use.

    Not using transformers pipelines on purpose: they pad each batch to its longest text, so almost
    every batch has a new shape, and MPS caches a compiled graph per shape until an 8 GB Mac runs out
    of memory. Padding to a multiple of PAD_MULTIPLE keeps the number of shapes small.
    fp32 everywhere: fp16 gave the same labels on MPS but was not faster.
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int | None = None,
        max_tokens: int | None = None,
        zero_shot_max_tokens: int | None = None,
    ) -> None:
        self.device = device or pick_device(settings.nlp_device)
        self.batch_size = batch_size or settings.nlp_batch_size
        self.max_tokens = max_tokens or settings.nlp_max_tokens
        self.zero_shot_max_tokens = zero_shot_max_tokens or settings.nlp_zero_shot_max_tokens
        self._models: dict[str, tuple[Any, Any]] = {}

    def _load(self, name: str) -> tuple[Any, Any]:
        if name not in self._models:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            log.info("loading %s on %s", name, self.device)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
            self._models[name] = (tokenizer, model)
        return self._models[name]

    def load_all(self) -> None:
        for name in ALL_MODELS:
            self._load(name)

    def _logits(
        self, name: str, first: list[str], second: list[str] | None, max_length: int, truncation: str | bool
    ) -> Any:
        """Logits for all inputs, computed in batches, returned as one CPU tensor."""
        import torch

        tokenizer, model = self._load(name)
        chunks = []
        for start in range(0, len(first), self.batch_size):
            pair = second[start : start + self.batch_size] if second is not None else None
            encoded = tokenizer(
                first[start : start + self.batch_size],
                pair,
                padding=True,
                pad_to_multiple_of=PAD_MULTIPLE,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                chunks.append(model(**encoded).logits.float().cpu())
        return torch.cat(chunks)

    def _classify(self, name: str, texts: list[str]) -> list[Scores]:
        if not texts:
            return []
        _, model = self._load(name)
        probs = self._logits(name, texts, None, self.max_tokens, True).softmax(dim=-1)
        labels = [str(model.config.id2label[i]).lower() for i in range(probs.shape[1])]
        return [dict(zip(labels, map(float, row))) for row in probs]

    def sentiment(self, texts: list[str]) -> list[Scores]:
        return self._classify(SENTIMENT_MODEL, texts)

    def emotion(self, texts: list[str]) -> list[Scores]:
        return self._classify(EMOTION_MODEL, texts)

    def zero_shot(
        self, texts: list[str], labels: list[str], template: str, multi_label: bool
    ) -> list[Scores]:
        """NLI zero-shot, same maths as transformers' zero-shot pipeline.

        Each (text, template.format(label)) pair is scored for entailment. Single-label: softmax of
        the entailment logits across labels. Multi-label: softmax of [contradiction, entailment]
        per pair, so each label gets an independent 0-1 score. Only the text is truncated.
        """
        if not texts or not labels:
            return []
        _, model = self._load(ZERO_SHOT_MODEL)
        label2id = {k.lower(): v for k, v in model.config.label2id.items()}
        entail, contra = label2id["entailment"], label2id["contradiction"]

        hypotheses = [template.format(label) for label in labels]
        premises = [t for t in texts for _ in labels]
        logits = self._logits(
            ZERO_SHOT_MODEL, premises, hypotheses * len(texts), self.zero_shot_max_tokens, "only_first"
        ).view(len(texts), len(labels), -1)

        if multi_label:
            scores = logits[..., [contra, entail]].softmax(dim=-1)[..., 1]
        else:
            scores = logits[..., entail].softmax(dim=-1)
        return [dict(zip(labels, map(float, row))) for row in scores]

    def release_memory(self) -> None:
        """Return cached GPU memory between batches; on an 8 GB Mac MPS otherwise runs out."""
        import torch

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def model_hashes(self) -> dict[str, str]:
        """Hub commit hash of each model, from its (small) config file."""
        from transformers import AutoConfig

        hashes = {}
        for name in ALL_MODELS:
            config = self._models[name][1].config if name in self._models else AutoConfig.from_pretrained(name)
            hashes[name] = str(getattr(config, "_commit_hash", None) or "unknown")
        return hashes
