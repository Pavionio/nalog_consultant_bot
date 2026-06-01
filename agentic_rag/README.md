# Agentic RAG (эксперимент)

Агентный RAG: LLM-агент (`openai/gpt-oss-20b` через OpenAI-совместимый LM Studio)
сам решает, что искать, и итеративно вызывает один инструмент — семантический
поиск по Qdrant. **Без реранкера** (только эмбеддер). Эксперимент изолирован в этой
папке и описан в `text/main.tex` (раздел «Эксперимент: агентный RAG»).

## Идея
В отличие от одношагового RAG (один запрос → top-k → ответ), агент **декомпозирует**
сложный вопрос и делает несколько последовательных поисков (multi-hop), например:
сначала норма НК РФ, затем разъяснение Минфина. Протокол — текстовый JSON-ReAct:
на каждом ходу модель возвращает ровно один JSON-объект:
```json
{"action": "search", "query": "...", "source_code": "<необязательно>"}
{"action": "answer", "content": "...с ссылками [n]..."}
```
Найденные фрагменты дедуплицируются и получают сквозные номера `[n]`. Финальный
ответ всегда регенерируется из собранного множества чанков через `build_context`,
чтобы нумерация и формат (`strict_citations`) были согласованы.

## Предварительные требования
1. **LM Studio** на Windows: запущен Local Server (порт 1234) и **загружена модель
   `openai/gpt-oss-20b`** (Load) либо включён JIT-loading; рекомендуется context length **32K**.
2. **Qdrant** запущен, коллекция `rag_chunks` присутствует (оригинальная, bge-m3).
3. GPU для эмбеддера bge-m3 (~1.3 ГБ; реранкер не грузится).

> **Сеть (WSL → Windows).** Клиент полностью обходит прокси (`session.trust_env=False`
> + `proxies=None`), иначе активный `http_proxy` перехватывает запросы (это давало
> ложные HTTP 503). Адрес по умолчанию `http://192.168.1.8:1234`; альтернатива —
> WSL-шлюз `http://172.18.96.1:1234` (флаг `--base-url`).

## Коллекция и эмбеддер
Эксперимент работает на **`rag_chunks` + `BAAI/bge-m3`** (а не на оптимальной
e5-large-коллекции бота), потому что «золотая» разметка eval-датасетов и single-shot
baseline из отчёта построены именно на `rag_chunks`. Это делает сравнение
agentic-vs-single-shot корректным. Переопределяется флагами `--collection` / `--embed-model`.

## Запуск
Из корня репозитория.

**Один вопрос (трейс + ответ):**
```bash
uv run python -m agentic_rag.run_agentic_rag \
  --question "ИП на УСН без работников: срок уплаты взносов за себя и можно ли уменьшить налог?"
# опции: --base-url, --model, --reasoning-effort {off,low,medium,high},
#        --max-iters 6, --per-call-top-k 4, --snippet-chars 600, --json-out
```

**Количественная оценка (agentic union vs single-shot dense, подмножество superhard):**
```bash
uv run python -m agentic_rag.eval_agentic_rag --limit 40 --seed 42 \
  > agentic_rag/agentic_eval_report.json 2> agentic_rag/agentic_eval_progress.log
```

## Параметры по умолчанию
| Параметр | Значение | Зачем |
|---|---|---|
| `temperature` | 0 | детерминизм (как везде в проекте) |
| `reasoning_effort` | low | как в judge/rewrite; держит бюджет контекста и latency |
| `max_iterations` | 6 | потолок числа поисков |
| `per_call_top_k` | 4 | фрагментов на один поиск |
| `snippet_chars` | 600 | усечение фрагмента в observation |
| `max_unique_chunks` | 24 | потолок накопленных уникальных чанков |

## Бюджет контекста
При лимитах выше накопленный тред упирается в ~10–14K токенов — помещается в 25K с
запасом. При `reasoning_effort=medium/high` или больших top-k держите контекст **32K**.

## Метрика оценки
`union Hit@k` — нашёл ли агент gold-документ среди **всех** своих поисков (объединение),
против single-shot Hit@k (один поиск top-5). Это **верхняя граница покрытия** (было ли
доказательство извлечено вообще), а НЕ ранжирующая метрика — напрямую не сравнима с
single-shot Hit@k как ранжирование. Матчинг переиспользует `_match_target`/`_match_doc`
из `src/eval/eval.py`.

## Файлы
| Файл | Назначение |
|---|---|
| `llm_client.py` | `GPTOSSChatClient` (OpenAI-совместимый, обход прокси, reasoning_effort) |
| `search_tool.py` | `SearchTool` (dense Qdrant, без реранкера, фильтр source_code) |
| `prompts.py` | системный промпт агента (JSON-протокол), промпт финального ответа |
| `agent.py` | `AgenticRAG` — цикл search/answer, дедуп, лимиты |
| `run_agentic_rag.py` | CLI: один вопрос → трейс + ответ |
| `eval_agentic_rag.py` | CLI: agentic vs single-shot → таблица + JSON-отчёт |
