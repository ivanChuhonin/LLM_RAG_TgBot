# -*- coding: utf-8 -*-
"""Точка входа: строит базу знаний и запускает Telegram-бота."""
from __future__ import annotations

import logging

from bot import build_application
from config import retriever_cfg
from data_loader import load_dataset
from llm_client import OllamaClient
from retriever import SemanticDocumentDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_database() -> SemanticDocumentDatabase:
    logger.info("Загрузка датасета...")
    df = load_dataset()
    logger.info("Строк в датасете: %d", len(df))

    database = SemanticDocumentDatabase()
    logger.info("Построение/загрузка FAISS-индекса...")
    database.build_index(df, force_rebuild=retriever_cfg.rebuild_index)
    logger.info("Индекс готов: %d документов.", len(database.questions))
    return database


def main() -> None:
    database = build_database()
    llm_client = OllamaClient()
    application = build_application(database, llm_client)

    logger.info("Бот запускается (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
