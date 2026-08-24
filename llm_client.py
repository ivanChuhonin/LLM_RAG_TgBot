# -*- coding: utf-8 -*-
"""
Клиент для генерации ответа через Ollama вместо локальной загрузки
transformers-модели (Llama-2-7b-hf).

Работает в двух режимах, в зависимости от config.OllamaConfig.base_url:
 - локальный Ollama Desktop (http://localhost:11434) — без авторизации;
 - облачные модели ollama.com (https://ollama.com) — нужен OLLAMA_API_KEY,
   передаётся как Bearer-токен.

Используется /api/chat — Ollama сам применяет chat-template нужной модели,
поэтому не нужно вручную собирать промпт под конкретную архитектуру, как
раньше приходилось делать под Llama-2.
"""
from __future__ import annotations

from typing import List

import aiohttp

from config import OllamaConfig, ollama_cfg, retriever_cfg
from retriever import SearchResult


class OllamaClient:
    def __init__(self, cfg: OllamaConfig = ollama_cfg) -> None:
        self.cfg = cfg

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    async def generate_answer(self, question: str, contexts: List[SearchResult]) -> str:
        """Формирует промпт из вопроса пользователя + нескольких найденных
        похожих вопросов/ответов (RAG-контекст, top-k вместо одного лучшего
        совпадения) и просит модель дать связный финальный ответ.

        Уверенность решений передаётся модели явно: если даже лучший
        кандидат имеет низкое сходство с вопросом, модель просят честно
        сказать, что уверенной рекомендации нет, вместо того чтобы выдавать
        первый попавшийся результат как надёжный."""

        if contexts:
            context_block = "\n".join(
                f"{i}. \"{c.answer}\" (similarity to question: {c.score:.2f})"
                for i, c in enumerate(contexts, start=1)
            )
            if contexts[0].score >= retriever_cfg.similarity_threshold:
                confidence_note = (
                    "The reference answers below have good similarity to the "
                    "question — use them confidently to build your answer."
                )
            else:
                confidence_note = (
                    "IMPORTANT: the reference answers below have LOW similarity "
                    "to the question, so they may not actually be relevant. "
                    "If none of them genuinely fit the question, say honestly "
                    "that you don't have a confident recommendation and ask a "
                    "short clarifying question instead of forcing an answer."
                )
        else:
            context_block = "(no reference answers found)"
            confidence_note = (
                "No reference answers were found at all. Be honest that you "
                "don't have a confident recommendation and ask a short "
                "clarifying question instead of guessing."
            )

        system_prompt = (
            "You are a friendly movie recommendation assistant. "
            f"{confidence_note} "
            "When you do recommend a specific movie, wrap its exact title in "
            "double asterisks, like **Movie Title** — do this for the title "
            "only, nowhere else. Keep the answer concise (2-4 sentences) and "
            "answer in the same language as the user's question."
        )
        user_prompt = (
            f"User question: {question}\n"
            f"Reference answers, ordered by relevance:\n{context_block}\n\n"
            "Write the final answer to the user."
        )

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_k": self.cfg.top_k,
                "num_predict": self.cfg.num_predict,
            },
        }

        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=self._headers()) as resp:
                resp.raise_for_status()
                data = await resp.json()

        return data["message"]["content"].strip()