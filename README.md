# 🎬 Movie RAG Telegram Bot

Telegram-бот, который рекомендует фильмы по свободному текстовому запросу пользователя. Под капотом — классический RAG-конвейер: семантический поиск похожего вопроса в базе фильмовых диалогов (FAISS + sentence-transformers) и генерация связного финального ответа через LLM (Ollama).

## Как это работает

```
Пользователь → Telegram-бот → семантический поиск (FAISS) → топ-N похожих вопросов/ответов
                                                                        │
                                                                        ▼
                                                    LLM (Ollama) формирует финальный ответ,
                                                    учитывая уверенность найденных совпадений
```

1. Вопрос пользователя кодируется в эмбеддинг (`sentence-transformers`).
2. По базе из ~2M пар вопрос-ответ (датасет диалогов из фильмов) ищутся `top-k` ближайших по смыслу через `faiss.IndexFlatIP` (косинусное сходство).
3. Найденные кандидаты вместе с их сходством передаются в LLM. Если сходство низкое, модель явно предупреждается не выдумывать уверенный ответ, а честно сказать, что не уверена, и уточнить запрос.
4. Ответ модели форматируется (название фильма выделяется жирным) и отправляется пользователю в Telegram.

## Возможности

- 🔍 **Семантический поиск**, а не по ключевым словам — находит перефразированные и синонимичные запросы.
- 🧠 **RAG с учётом уверенности** — при слабом совпадении бот не выдаёт случайный ответ, а честно признаётся, что не уверен.
- ⚡ **Кэширование индекса** — FAISS-индекс строится один раз и переиспользуется между запусками.
- ☁️ **LLM через Ollama** — работает как с облачными моделями (ollama.com), так и с локальными (Ollama Desktop), переключается одной переменной окружения.
- 💬 **Понятный UX в Telegram** — приветствия/пустые сообщения обрабатываются подсказкой, а не уходят в LLM; названия фильмов выделяются жирным.

## Стек

| Компонент | Технология |
|---|---|
| Telegram-бот | `python-telegram-bot` (v20+, async) |
| Эмбеддинги | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Векторный поиск | `faiss` |
| Генерация ответа | Ollama (облако или локально) |
| Данные | Kaggle: *The Movie Dialog Dataset* |

## Установка

```bash
git clone <URL_репозитория>
cd llm_rag_tgbot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Заполните `.env`:

| Переменная | Описание |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Токен бота от [@BotFather](https://t.me/BotFather) |
| `OLLAMA_BASE_URL` | `https://ollama.com` (облако) или `http://localhost:11434` (локально) |
| `OLLAMA_API_KEY` | Нужен только для облачных моделей ([ollama.com/settings/keys](https://ollama.com/settings/keys)) |
| `OLLAMA_MODEL` | Например `gemma4:cloud` или `gemma4:12b` для локального запуска |
| `EMBEDDING_MODEL` | Модель эмбеддингов для FAISS-поиска |
| `SIMILARITY_THRESHOLD` | Порог уверенности (0–1) для рекомендаций |
| `TOP_N_CANDIDATES` | Сколько кандидатов передавать в LLM за раз |

## Данные

Файлы датасета не входят в репозиторий — получите их один раз через Kaggle CLI:

```bash
pip install kaggle
# положите kaggle.json (Kaggle → Account → Create New API Token) в ~/.kaggle/kaggle.json
kaggle datasets download -d abhishek/the-movie-dialog-dataset -f task3_qarecs_train.txt -p data --unzip
kaggle datasets download -d abhishek/the-movie-dialog-dataset -f task1_qa_train.txt -p data --unzip
```

Файлы должны оказаться в `./data/task3_qarecs_train.txt` и `./data/task1_qa_train.txt`.

## Запуск

Для облачной модели дополнительная установка не нужна — только ключ в `.env`. Для локальной модели через Ollama Desktop:

```bash
ollama pull gemma4:12b
ollama serve
```

Затем:

```bash
python main.py
```

При первом запуске построится FAISS-индекс (может занять время в зависимости от размера датасета и мощности CPU) и закэшируется в `index_cache/`. Повторные запуски стартуют мгновенно.

## Структура проекта

```
.
├── main.py           # точка входа
├── bot.py            # обработчики Telegram
├── llm_client.py      # клиент Ollama, промпт-инжиниринг и учёт уверенности
├── retriever.py        # FAISS + sentence-transformers поиск
├── data_loader.py      # загрузка/очистка датасета
├── config.py           # конфигурация из .env
├── requirements.txt
├── .env.example
└── data/                # файлы датасета (не в репозитории)
```

## Лицензия

MIT
