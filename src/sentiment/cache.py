"""
SQLite-backed sentiment cache. Survives server restarts.
Falls back to in-memory if the DB file can't be created.
"""
import sqlite3
import json
import time
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_cache.db')

_DDL = '''
CREATE TABLE IF NOT EXISTS sentiment_cache (
    cache_key  TEXT PRIMARY KEY,
    result     TEXT NOT NULL,
    expires_at REAL NOT NULL
)
'''

# In-memory fallback if SQLite unavailable
_mem: dict = {}


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute(_DDL)
    c.commit()
    return c


def get(key: str):
    try:
        with _conn() as c:
            row = c.execute(
                'SELECT result, expires_at FROM sentiment_cache WHERE cache_key=?', (key,)
            ).fetchone()
        if row and time.time() < row[1]:
            return json.loads(row[0])
    except Exception:
        # Fallback to in-memory
        if key in _mem:
            val, exp = _mem[key]
            if time.time() < exp:
                return val
    return None


def set(key: str, result: dict, ttl: int = 600):
    try:
        with _conn() as c:
            c.execute(
                'INSERT OR REPLACE INTO sentiment_cache VALUES (?,?,?)',
                (key, json.dumps(result), time.time() + ttl)
            )
    except Exception:
        _mem[key] = (result, time.time() + ttl)


def evict_expired():
    """Remove stale rows — call occasionally to keep DB lean."""
    try:
        with _conn() as c:
            c.execute('DELETE FROM sentiment_cache WHERE expires_at < ?', (time.time(),))
    except Exception:
        pass
