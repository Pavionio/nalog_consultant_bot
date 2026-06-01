# Налоговый консультант — RAG чат-бот

Telegram-бот, который отвечает на вопросы по налогам МСП на основе официальных
документов ФНС России, Минфина России и `pravo.gov.ru`. Использует RAG: находит
релевантные фрагменты нормативных документов и формирует ответ со ссылками на
источники, отказываясь отвечать при недостатке данных.

Оптимальный пайплайн (по результатам экспериментов):
**e5-large** (эмбеддинги) → **dense-поиск в Qdrant** → **реранкер BGE-v2-m3** →
сборка контекста → **Qwen3-14B** (llama.cpp), `strict_citations`, `temperature=0`.

Подробности архитектуры и данных — в [docs/design.md](docs/design.md); экспериментальное
обоснование выбора компонентов — в отчёте `text/main.tex`.

## Запуск

Бот и сервисы поднимаются через Docker Compose. Инструкция предполагает, что
**данные уже подготовлены** — том Qdrant содержит коллекцию с эмбеддингами
(`QDRANT_COLLECTION`). Подготовка корпуса с нуля описана в [docs/design.md](docs/design.md).

### 1. Предварительные требования
* Docker + NVIDIA Container Toolkit (нужен GPU — llama.cpp, эмбеддер и реранкер работают на CUDA);
* GPU с ≈16 ГБ VRAM (Qwen3-14B q4 + e5-large + реранкер).

### 2. Файл `.env` (в корне репозитория)
```
HF_TOKEN=<токен Hugging Face>
TELEGRAM_BOT_TOKEN=<токен Telegram-бота от @BotFather>
LLM=Qwen/Qwen3-14B-GGUF:q4_k_m
QDRANT_COLLECTION=rag_chunks_e5_large_token_1024_256_clean_no_boilerplate
EMBED_MODEL=intfloat/multilingual-e5-large
EMBED_DEVICE=cuda
RAG_USE_RERANKER=true
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANKER_FETCH_K=50
```

### 3. Запуск контейнеров
```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
```
Поднимутся четыре сервиса: `postgres`, `qdrant`, `llamacpp` и `bot`.
`--env-file .env` нужен, чтобы compose подставил модель `${LLM}` в команду llama.cpp.

При первом запуске будут скачаны веса: модель LLM (~9 ГБ, в том `llama_cache`),
эмбеддер и реранкер (~3.5 ГБ, в том `hf_cache`) — это занимает несколько минут.
Образ бота тоже собирается при первом `up` (тянет torch + зависимости).

### 4. Проверка и управление
```bash
docker compose -f docker/docker-compose.yml logs -f bot     # логи бота
docker compose -f docker/docker-compose.yml ps              # статус сервисов
docker compose -f docker/docker-compose.yml --env-file .env down   # остановить
```

После старта напишите боту в Telegram `/start`, примите правила (`/accept`) и
задайте налоговый вопрос обычным сообщением.

### Команды бота
`/start` — приветствие и правила · `/accept` — принять правила ·
`/rules` — правила · `/reset`, `/new_sesssion` — сбросить историю · `/help` — справка.

## Структура репозитория
| Путь | Назначение |
|---|---|
| `src/bot/` | Telegram-бот (aiogram) и адаптер к RAG |
| `src/rag/` | RAG-пайплайн: эмбеддеры, реранкеры, чанкинг, очистка, генерация |
| `src/eval/`, `scripts/` | оценка качества и эксперименты |
| `fetch/` | сбор и индексация документов из источников |
| `docker/` | Compose-файл и образ бота |
| `docs/` | проектная документация (`design.md` и др.) |
| `text/` | отчёт о проекте (LaTeX) |
