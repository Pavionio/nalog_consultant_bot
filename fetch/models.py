from dataclasses import dataclass
from typing import Optional

@dataclass
class Source:
    code: str
    base_url: str
    kind: str
    active: bool
    handler: str

@dataclass
class DiscoveredDoc:
    source_code: str
    url: str
    external_id: str
    kind: str

# опционально для будущего “обновления”
@dataclass
class FetchResult:
    status_code: int
    content: bytes
    etag: Optional[str] = None
    last_modified: Optional[str] = None