"""
Schema helpers for media assets storage (SQLite).
"""

MEDIA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS media_assets (
    id TEXT PRIMARY KEY,              -- content hash
    phash TEXT,
    parent_id TEXT,
    faces TEXT,                       -- JSON list of clusters
    brands TEXT,                      -- JSON list of strings
    tech_meta TEXT,                   -- JSON blob: resolution/bitrate/fps
    created_at REAL,
    path TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_phash ON media_assets(phash);
"""
