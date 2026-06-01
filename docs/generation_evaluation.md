# Оценка качества генерации

Этот пайплайн оценивает не поиск сам по себе, а качество ответа, который модель сгенерировала по уже найденным чанкам. Входной файл содержит вопрос, найденный контекст, ответ модели и ручные оценки, если они уже проставлены.

## Метрики ответа

**Precision** здесь означает фактическую точность генерации, а не retrieval Precision@K. Валидатор проверяет, подтверждаются ли утверждения ответа найденным контекстом: сроки, суммы, обязанности, исключения, реквизиты, номера статей и выводы. Основной источник истины — чанки `gold_relevant`. Чанки `hard_negative` похожи на правильные, но нерелевантны для вопроса, поэтому их нельзя использовать как доказательство. Галлюцинации и неподтверждённая конкретика снижают `precision`.

**Completeness** проверяет полноту: покрыл ли ответ важные аспекты из `gold_relevant` chunks. К важным аспектам относятся условия, исключения, сроки, категории налогоплательщиков, налоговый режим, период, реквизиты и ограничения. Если ответ даёт общий вывод, но пропускает важное условие из релевантного контекста, оценка снижается.

**Format** оценивает форму ответа: читаемость, структуру, профессиональный тон, отсутствие грубости, битой кодировки, сырого HTML, мусорной Markdown/LaTeX-разметки, а также соблюдение требуемого формата. Для `strict_citations` учитывается наличие блока источников, если источники были доступны и требовались.

Шкала 1–5 используется для всех трёх метрик, потому что она достаточно детальна для различения грубых ошибок, частичных ответов и хороших ответов, но остаётся удобной для ручной разметки.

## Human scores

По умолчанию ручные оценки равны `-1`:

```json
"human": {
  "precision": -1,
  "completeness": -1,
  "format": -1,
  "comment": ""
}
```

`-1` означает, что человек ещё не оценил пример. Такие значения не участвуют в согласованности человек-судья и Каппе Коэна. LLM-судью при этом можно запускать и сохранять независимо.

## Согласованность

Каппа Коэна показывает согласованность человека и LLM-судьи с поправкой на случайные совпадения.

Интерпретация:

| Каппа | Интерпретация |
|---:|---|
| < 0 | хуже случайного совпадения |
| 0.00–0.20 | слабое согласие |
| 0.21–0.40 | удовлетворительное согласие |
| 0.41–0.60 | умеренное согласие |
| 0.61–0.80 | существенное согласие |
| 0.81–1.00 | почти полное согласие |

Так как шкала 1–5 порядковая, дополнительно считаются линейная и квадратичная взвешенная Каппа. Ошибка `5 -> 4` меньше, чем `5 -> 1`, и weighted kappa учитывает это.

Дополнительные метрики:

- `exact_agreement`: доля примеров, где `human score == judge score`.
- `within_1_agreement`: доля примеров, где `abs(human - judge) <= 1`.
- `MAE`: средняя абсолютная ошибка между human и judge.

## Почему три валидатора

`precision`, `completeness` и `format` — разные свойства ответа. Один общий судья часто смешивает фактическую корректность, полноту и стиль: красивый, но неверный ответ может получить завышенную оценку, а точный короткий ответ — заниженную. Три отдельных валидатора дают более стабильную и интерпретируемую картину.

## Команды

Сгенерировать ответы:

```bash
uv run python scripts/build_generation_eval_dataset.py \
  --dataset eval_hard_dataset.jsonl \
  --out data/metrics/generation_eval_hard_reranker.jsonl \
  --method reranker \
  --qdrant-collection <BEST_COLLECTION> \
  --embed-model <BEST_EMBEDDER> \
  --top-k 5 \
  --reranker-model <BEST_RERANKER> \
  --reranker-fetch-k 50 \
  --generator-model <GENERATOR_MODEL> \
  --generator-base-url <GENERATOR_BASE_URL> \
  --prompt-variant strict_citations \
  --limit 50
```

`--generator-model` задаёт модель, которая пишет ответы. `--judge-model` задаёт модель, которая оценивает ответы. Они могут быть разными; желательно, чтобы модель-судья была не слабее модели генерации.

Экспортировать CSV для ручной оценки:

```bash
uv run python scripts/export_generation_for_human_review.py \
  --input data/metrics/generation_eval_hard_reranker.jsonl \
  --out data/metrics/generation_eval_hard_reranker_review.csv
```

Импортировать ручные оценки:

```bash
uv run python scripts/import_human_generation_scores.py \
  --jsonl data/metrics/generation_eval_hard_reranker.jsonl \
  --csv data/metrics/generation_eval_hard_reranker_review.csv \
  --out data/metrics/generation_eval_hard_reranker_human.jsonl
```

Запустить LLM-судей:

```bash
uv run python scripts/evaluate_generation_judges.py \
  --input data/metrics/generation_eval_hard_reranker_human.jsonl \
  --out data/metrics/generation_eval_hard_reranker_judged.jsonl \
  --summary-out data/metrics/generation_eval_hard_reranker_summary.json \
  --judge-model <JUDGE_MODEL> \
  --judge-base-url <JUDGE_BASE_URL> \
  --validators precision completeness format \
  --judge-examples on
```

Если ручная разметка идёт параллельно с LLM-судьёй, можно не ждать человека перед запуском валидаторов:

```bash
uv run python scripts/evaluate_generation_judges.py \
  --input data/metrics/generation_eval_hard_reranker.jsonl \
  --out data/metrics/generation_eval_hard_reranker_judged.jsonl \
  --summary-out data/metrics/generation_eval_hard_reranker_judge_only_summary.json \
  --judge-model <JUDGE_MODEL> \
  --judge-base-url <JUDGE_BASE_URL>
```

Пока судья работает, человек размечает CSV и импортирует оценки в отдельный JSONL. После этого отдельный compare-скрипт объединяет `judge` из judged-файла и `human` из human-файла, а затем считает agreement и Каппу Коэна без повторного вызова LLM:

```bash
uv run python scripts/compare_generation_human_vs_judge.py \
  --judged-jsonl data/metrics/generation_eval_hard_reranker_judged.jsonl \
  --human-jsonl data/metrics/generation_eval_hard_reranker_human.jsonl \
  --out data/metrics/generation_eval_hard_reranker_judged_human.jsonl \
  --summary-out data/metrics/generation_eval_hard_reranker_human_vs_judge_summary.json
```

Запустить только один валидатор:

```bash
uv run python scripts/evaluate_generation_judges.py \
  --input data/metrics/generation_eval_hard_reranker_human.jsonl \
  --out data/metrics/generation_eval_hard_reranker_precision_judged.jsonl \
  --summary-out data/metrics/generation_eval_hard_reranker_precision_summary.json \
  --judge-model <JUDGE_MODEL> \
  --judge-base-url <JUDGE_BASE_URL> \
  --validators precision
```

## Использование в отчёте

В отчёте по generation quality обычно показывают:

- средние оценки LLM-судьи по `precision`, `completeness`, `format`;
- средние ручные оценки, если они есть;
- exact agreement, ±1 agreement, MAE и Каппу Коэна;
- долю отказов, когда контекст не содержит ответа;
- качество найденного контекста: сколько `gold_relevant`, `hard_negative`, `unknown` chunks попало в top-k;
- дефекты формата: нечитаемость, HTML, сломанная разметка, непрофессиональный тон.
