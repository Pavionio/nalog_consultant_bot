CREATE TABLE IF NOT EXISTS rag_crawl_task (
  id              bigserial PRIMARY KEY,
  doc_id          bigint NOT NULL REFERENCES rag_doc(id) ON DELETE CASCADE,

  -- что сделать
  task_type       text NOT NULL DEFAULT 'fetch', -- fetch|reindex|discover (можешь расширять)
  reason          text NOT NULL,                  -- new|changed|periodic|manual

  -- планирование
  not_before      timestamptz NOT NULL DEFAULT now(), -- не запускать раньше этого времени
  priority        int NOT NULL DEFAULT 100,            -- меньше = важнее (0..1000)

  -- исполнение
  status          text NOT NULL DEFAULT 'queued', -- queued|running|done|failed
  attempts        int NOT NULL DEFAULT 0,
  max_attempts    int NOT NULL DEFAULT 5,
  locked_at       timestamptz,
  lock_owner      text,                             -- имя воркера/хоста

  -- диагностика
  last_error      text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_task_pick
ON rag_crawl_task(status, not_before, priority, id);

-- запретить две активные задачи на один doc_id одновременно (чтобы не было гонок)
CREATE UNIQUE INDEX IF NOT EXISTS uq_task_one_active_per_doc
ON rag_crawl_task(doc_id)
WHERE status IN ('queued','running');
