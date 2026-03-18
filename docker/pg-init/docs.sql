CREATE TABLE IF NOT EXISTS rag_doc (
  id                bigserial PRIMARY KEY,

  -- идентификация документа
  source_code       text NOT NULL,            -- код источника из configs/sources.yaml (nalog_letters, minfin_answers...)
  external_id       text,                     -- id документа у источника (если есть); может быть NULL
  canonical_url     text NOT NULL,            -- каноническая ссылка на документ

  -- тип и базовые метаданные (полезны для цитирования)
  kind              text NOT NULL,            -- 'law' | 'order' | 'letter' | 'calendar' | ...
  title             text,
  published_at      date,                     -- дата публикации на сайте/в ленте
  doc_date          date,                     -- дата самого документа (например "от 01.02.2026")
  doc_number        text,                     -- номер документа (если есть)

  -- обнаружение и жизненный цикл
  status            text NOT NULL DEFAULT 'active', -- active/withdrawn/archived/unknown
  last_seen_at      timestamptz NOT NULL DEFAULT now(), -- когда последний раз увидели в листинге/сайте

  -- обновления: когда и как проверять
  next_check_at     timestamptz,              -- когда планово проверять снова (по crawl_freq)
  last_fetch_at     timestamptz,              -- когда реально последний раз скачали контент

  -- HTTP-валидаторы для экономного обновления
  http_etag         text,                     -- ETag с сервера (если отдаёт)
  http_last_mod     text,                     -- Last-Modified с сервера (если отдаёт)

  -- контроль изменений контента
  content_sha256    text,                     -- sha256 нормализованного текста/контента (для сравнения)

  -- ошибки по документу (в дополнение к task)
  error_count       int NOT NULL DEFAULT 0,   -- сколько ошибок подряд при обработке
  last_error        text,                     -- последняя ошибка (коротко)

  -- связь с Qdrant (сам текст/чанки хранятся там)
  qdrant_collection text NOT NULL DEFAULT 'rag_chunks',
  qdrant_doc_key    text,                     -- стабильный ключ документа (например "nalog_letters:12345" или hash(url))
  qdrant_revision   int  NOT NULL DEFAULT 0,  -- увеличивается при изменении контента
  indexed_at        timestamptz,              -- когда текущая revision успешно залита в Qdrant

  -- локальные пути к файлам (для отладки)
  raw_path          text,
  text_path         text,
  chunks_path       text
);

-- уникальность "документа" внутри источника (если external_id существует)
CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_doc_source_external
ON rag_doc(source_code, external_id)
WHERE external_id IS NOT NULL;

-- чтобы не плодить одинаковые URL (если external_id нет)
CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_doc_canonical_url
ON rag_doc(canonical_url);

CREATE INDEX IF NOT EXISTS idx_rag_doc_next_check
ON rag_doc(next_check_at);

CREATE INDEX IF NOT EXISTS idx_rag_doc_source
ON rag_doc(source_code);

CREATE INDEX IF NOT EXISTS idx_rag_doc_seen
ON rag_doc(last_seen_at);


