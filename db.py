"""
db.py — Database configuration for SkyIQ
Supports both SQLite (development) and MySQL (production)

Set DB_TYPE in config.py to switch between them:
    DB_TYPE = 'sqlite'   ← default, works on Termux instantly
    DB_TYPE = 'mysql'    ← use when MySQL server is running
"""

import os, sqlite3

# ── LOAD CONFIG ────────────────────────────────────────────
try:
    from config import DB_TYPE, MYSQL_CONFIG
except ImportError:
    DB_TYPE      = 'sqlite'
    MYSQL_CONFIG = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'database', 'flights.db')


# ══════════════════════════════════════════════════════════
# CONNECTION FACTORY
# ══════════════════════════════════════════════════════════
def get_db():
    """
    Returns a database connection.
    Automatically uses SQLite or MySQL based on config.py
    Both return a connection with dict-like row access.
    """
    if DB_TYPE == 'mysql':
        return _get_mysql()
    return _get_sqlite()


def _get_sqlite():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_mysql():
    try:
        import pymysql
        import pymysql.cursors
        conn = pymysql.connect(
            host     = MYSQL_CONFIG.get('host',     'localhost'),
            user     = MYSQL_CONFIG.get('user',     'skyiq'),
            password = MYSQL_CONFIG.get('password', 'skyiq123'),
            database = MYSQL_CONFIG.get('database', 'flight_delay_db'),
            charset  = 'utf8mb4',
            cursorclass = pymysql.cursors.DictCursor,
            autocommit  = False
        )
        return conn
    except ImportError:
        raise RuntimeError("pymysql not installed. Run: pip install pymysql")
    except Exception as e:
        raise RuntimeError(f"MySQL connection failed: {e}\n"
                           f"Make sure mysqld_safe is running.")


# ══════════════════════════════════════════════════════════
# QUERY HELPER
# Abstracts SQLite (?) vs MySQL (%s) placeholder difference
# ══════════════════════════════════════════════════════════
def query(conn, sql, params=None):
    """
    Execute a SELECT query and return all rows as list of dicts.
    Handles SQLite vs MySQL placeholder syntax automatically.
    """
    sql    = _fix_sql(sql)
    params = params or ()
    if DB_TYPE == 'mysql':
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    else:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(conn, sql, params=None):
    """
    Execute a SELECT query and return first row as dict, or None.
    """
    rows = query(conn, sql, params)
    return rows[0] if rows else None


def execute(conn, sql, params=None):
    """
    Execute INSERT / UPDATE / DELETE.
    Returns lastrowid.
    """
    sql    = _fix_sql(sql)
    params = params or ()
    if DB_TYPE == 'mysql':
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid
    else:
        cur = conn.execute(sql, params)
        return cur.lastrowid


def commit(conn):
    conn.commit()


def close(conn):
    conn.close()


def _fix_sql(sql):
    """Convert SQLite ? placeholders to MySQL %s placeholders."""
    if DB_TYPE == 'mysql':
        return sql.replace('?', '%s')
    return sql
