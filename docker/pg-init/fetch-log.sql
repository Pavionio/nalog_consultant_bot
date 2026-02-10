CREATE TABLE IF NOT EXISTS rag_fetch_log (
  id            bigserial PRIMARY KEY,
  doc_id        bigint REFERENCES rag_doc(id) ON DELETE SET NULL,

  url           text NOT NULL,
  fetched_at    timestamptz NOT NULL DEFAULT now(),

  method        text NOT NULL DEFAULT 'GET',   -- GET/HEAD
  status_code   int,                          -- 200/304/404/...
  elapsed_ms    int,                          -- сколько заняло (для мониторинга)

  -- что отправляли/что получили (по минимуму)
  req_if_none_match   text,                   -- If-None-Match
  req_if_mod_since    text,                   -- If-Modified-Since
  resp_etag           text,                   -- ETag
  resp_last_mod       text,                   -- Last-Modified

  body_bytes    bigint,
  body_sha256   text,                         -- sha256 сырых байт (или нормализованного текста — на твой выбор)
  error         text
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_doc_time
ON rag_fetch_log(doc_id, fetched_at DESC);
