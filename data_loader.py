# -*- coding: utf-8 -*-
"""
Загрузка датасета вопрос-ответ.

    pip install kaggle
    # положите ваш kaggle.json в ~/.kaggle/kaggle.json (chmod 600)
    kaggle datasets download -d abhishek/the-movie-dialog-dataset \
        -f task3_qarecs_train.txt -p data --unzip
    kaggle datasets download -d abhishek/the-movie-dialog-dataset \
        -f task1_qa_train.txt -p data --unzip

Если файлов нет, initialize_dataset() бросит понятную ошибку с этой же
подсказкой.
"""
from __future__ import annotations

import string
from pathlib import Path

import pandas as pd

from config import DATA_DIR

REQUIRED_FILES = ["task3_qarecs_train.txt", "task1_qa_train.txt"]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _remove_punctuation(series: pd.Series) -> pd.Series:
    return series.str.translate(_PUNCT_TABLE)


def _check_files_present() -> None:
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Не найдены файлы датасета: "
            f"{missing} в папке {DATA_DIR}.\n"
            "Скачайте их один раз через Kaggle CLI, см. docstring "
            "data_loader.py, и положите в ./data."
        )


def load_dataset() -> pd.DataFrame:
    """Загружает и объединяет датасет вопрос-ответ, как в исходном ноутбуке."""
    _check_files_present()

    df = pd.read_csv(
        DATA_DIR / "task3_qarecs_train.txt", sep="\t", names=["text", "answer"]
    )
    df["text"] = df["text"].str[2:]

    df_extra = pd.read_csv(
        DATA_DIR / "task1_qa_train.txt", sep="\t", names=["text", "answer"]
    )
    df_extra["text"] = df_extra["text"].str[2:]

    # Каждая третья строка в исходном train-файле — служебная (как в ноутбуке).
    mask = [i % 3 != 1 for i in range(len(df))]
    df = df[mask].copy()

    df["text"] = _remove_punctuation(df["text"])
    df_extra["text"] = _remove_punctuation(df_extra["text"])

    df = pd.concat([df, df_extra], ignore_index=True)
    df = df.dropna(subset=["text", "answer"]).reset_index(drop=True)
    df = df.head(1000000)
    return df


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Загружено строк: {len(dataset)}")
    print(dataset.head(5))
