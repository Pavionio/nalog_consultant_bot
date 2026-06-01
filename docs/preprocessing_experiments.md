# Эксперименты с предобработкой данных

## Зачем нужна предобработка

Официальные сайты часто содержат навигацию, футеры, cookie-баннеры и HTML-артефакты. Такой шум ухудшает чанкирование и embedding, а значит и качество retrieval.

Слишком агрессивная очистка тоже опасна: можно удалить юридически значимые реквизиты (даты, номера документов, статьи и пункты), что снижает точность поиска и повышает риск неверных ответов.

## Профили очистки

| Profile | Что делает | Риск |
|---|---|---|
| raw | без очистки, только базовый `strip` | много шума |
| clean_basic | нормализация пробелов/переносов, HTML entities, управляющие символы | низкий риск |
| clean_legal | `clean_basic` + удаление навигационного мусора с защитой юридических строк | средний риск |
| clean_aggressive | `clean_legal` + удаление повторов и короткого/пунктуационного шума | риск удалить полезное |
| clean_no_boilerplate | `clean_legal` + удаление corpus-level boilerplate (часто повторяющихся строк) | риск удалить повторяющиеся юридические фразы |

## Что считаем юридически значимым

- даты (`01.01.2024`, `1 января 2024`, `от 12.03.2024`);
- номера документов (`№`, `N`, номера писем/приказов);
- ссылки на статьи/пункты/подпункты (`статья`, `ст.`, `пункт`, `подпункт`);
- ссылки на НК РФ и федеральные акты;
- налоговые термины (`НДФЛ`, `НДС`, `УСН`, `НПД`, `ПСН`, `ЕНВД`, `налогоплательщик`, `налоговый агент`, и т.д.).

## Метрики

- **Hit@5 / Recall@5**: найден ли релевантный документ/чанк в top-5.
- **Precision@5**: доля релевантных результатов в top-5.
- **MRR@5**: насколько высоко находится первый релевантный результат.
- **nDCG@5**: качество ранжирования с учетом позиции.
- **HardNegativeRate@5**: как часто в top-5 попадают похожие, но неверные документы.
- **Avg removed char ratio**: насколько агрессивно профиль удаляет текст.
- **Docs with warnings**: на скольких документах очистка выглядит рискованной.
- **Index points count**: влияние очистки на размер индекса.
- **Latency (dense/rerank/p95)**: влияние очистки на скорость retrieval.

## Как запускать

Dry run:

```bash
uv run python scripts/run_preprocessing_experiments.py \
  --dry-run \
  --profiles raw clean_basic clean_legal \
  --datasets eval_hard_dataset.jsonl \
  --methods baseline \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128
```

Smoke reindex одного профиля:

```bash
uv run python scripts/reindex_local_corpus.py \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection rag_chunks_bge_m3_token1024_clean_legal_smoke \
  --embed-model BAAI/bge-m3 \
  --embed-device cuda \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --cleaning-profile clean_legal \
  --max-docs 30 \
  --recreate-collection \
  --save-cleaned-preview \
  --no-network
```

Smoke eval:

```bash
uv run python -m src.eval.eval \
  --dataset eval_hard_dataset.jsonl \
  --k 5 \
  --no-judge \
  --qdrant-collection rag_chunks_bge_m3_token1024_clean_legal_smoke \
  --embed-model BAAI/bge-m3 \
  --model clean_legal_smoke
```

Stage 1 (без reranker):

```bash
uv run python scripts/run_preprocessing_experiments.py \
  --profiles raw clean_basic clean_legal clean_aggressive clean_no_boilerplate \
  --datasets eval_hard_dataset.jsonl \
  --methods baseline \
  --k 5 \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --embed-model BAAI/bge-m3 \
  --embed-device cuda \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --skip-existing
```

Full experiment:

```bash
uv run python scripts/run_preprocessing_experiments.py \
  --profiles raw clean_basic clean_legal clean_aggressive clean_no_boilerplate \
  --datasets eval_dataset.jsonl eval_hard_dataset.jsonl eval_superhard_dataset.jsonl \
  --methods baseline reranker \
  --k 5 \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --embed-model <BEST_EMBEDDER> \
  --embed-device cuda \
  --chunk-method <BEST_CHUNK_METHOD> \
  --chunk-size <BEST_CHUNK_SIZE> \
  --chunk-overlap <BEST_CHUNK_OVERLAP> \
  --reranker-model <BEST_RERANKER> \
  --reranker-fetch-k <BEST_FETCH_K> \
  --skip-existing
```

Parent-child вариант:

```bash
uv run python scripts/run_preprocessing_experiments.py \
  --profiles raw clean_basic clean_legal clean_aggressive clean_no_boilerplate \
  --datasets eval_dataset.jsonl eval_hard_dataset.jsonl eval_superhard_dataset.jsonl \
  --methods baseline reranker \
  --k 5 \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --embed-model <BEST_EMBEDDER> \
  --embed-device cuda \
  --chunk-method parent_child \
  --parent-chunk-size 3072 \
  --parent-chunk-overlap 256 \
  --child-chunk-size 768 \
  --child-chunk-overlap 96 \
  --parent-chunker-method recursive_legal \
  --child-chunker-method sentence \
  --reranker-model <BEST_RERANKER> \
  --reranker-fetch-k <BEST_FETCH_K> \
  --skip-existing
```

## Как интерпретировать результаты

- если `clean_basic` улучшает метрики и не увеличивает hard negatives — это безопасное улучшение;
- если `clean_aggressive` поднимает Hit@5, но растит warnings или hard negatives — использовать осторожно;
- если `raw` не хуже clean-профилей — текущий extract уже достаточно чистый;
- если `clean_no_boilerplate` заметно уменьшает индекс без просадки метрик — хороший production-кандидат.
