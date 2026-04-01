"""
config.py — SkyIQ Configuration
Change DB_TYPE here to switch between SQLite and MySQL.
"""

# ── DATABASE TYPE ───────────────────────────────────────────
# Options: 'sqlite' or 'mysql'
DB_TYPE = 'sqlite'   # ← change to 'mysql' when MySQL is ready

# ── MYSQL CONFIG (only used when DB_TYPE = 'mysql') ─────────
MYSQL_CONFIG = {
    'host':     'localhost',
    'user':     'skyiq',
    'password': 'skyiq123',
    'database': 'flight_delay_db',
}

# ── FLASK CONFIG ────────────────────────────────────────────
SECRET_KEY = 'skyiq-secret-change-in-production'
DEBUG      = True
PORT       = 5000
