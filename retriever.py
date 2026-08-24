# -*- coding: utf-8 -*-
"""
Семантический поиск похожего вопроса в базе через эмбеддинги + FAISS.

Почему не TF-IDF:
 - TF-IDF совпадает по словам, а не по смыслу — "recommend a movie about war"
   и "any films set during a war" почти не пересекутся по токенам.
 - Эмбеддинги sentence-transformers кодируют смысл фразы целиком, что для
   вопросов пользователя (перефразировки, синонимы) существенно точнее.
 - FAISS (IndexFlatIP на нормированных векторах = косинусное сходство)
   ищет по сотням тысяч векторов практически мгновенно и легко
   сохраняется/загружается с диска, не требуя пересчёта при каждом запуске.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, retriever_cfg


@dataclass
class SearchResult:
    question: str
    answer: str
    score: float


class SemanticDocumentDatabase:
    """Хранит пары (вопрос, ответ) и ищет ближайший вопрос по смыслу."""

    def __init__(
        self,
        embedding_model_name: str = retriever_cfg.embedding_model,
        cache_dir: Path = INDEX_DIR,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        self._model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.answers: List[str] = []
        self.questions: List[str] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _index_paths(self) -> tuple[Path, Path]:
        faiss_path = self.cache_dir / "documents.faiss"
        meta_path = self.cache_dir / "documents_meta.pkl"
        return faiss_path, meta_path

    def build_index(self, df: pd.DataFrame, force_rebuild: bool = False) -> None:
        """Строит (или загружает из кэша) FAISS-индекс по колонке 'text'."""
        faiss_path, meta_path = self._index_paths()

        if not force_rebuild and faiss_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(faiss_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.questions = meta["questions"]
            self.answers = meta["answers"]
            return

        self.questions = df["text"].astype(str).tolist()
        self.answers = df["answer"].astype(str).tolist()

        embeddings = self.model.encode(
            self.questions,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # для косинусного сходства через IP
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"questions": self.questions, "answers": self.answers}, f)

    def search_top_k(
        self, query: str, k: int = retriever_cfg.top_n_candidates
    ) -> List[SearchResult]:
        """Возвращает до k ближайших по смыслу вопросов/ответов, отсортированных
        по убыванию сходства. Порог здесь не применяется — решение о том,
        насколько доверять результатам, принимается на уровне промпта
        (см. llm_client.generate_answer), чтобы модель могла честно сказать
        "не уверен", а не молчать при отсутствии единственного "хорошего"
        совпадения."""
        if self.index is None:
            raise RuntimeError("Индекс не построен — вызовите build_index() сначала.")

        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(query_vec, k=k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    question=self.questions[int(idx)],
                    answer=self.answers[int(idx)],
                    score=float(score),
                )
            )
        return results

    def search(
        self, query: str, threshold: float = retriever_cfg.similarity_threshold
    ) -> Optional[SearchResult]:
        """Обратная совместимость: только лучший результат, отфильтрованный
        по порогу. Новый код должен использовать search_top_k()."""
        results = self.search_top_k(query, k=1)
        if not results or results[0].score < threshold:
            return None
        return results[0]