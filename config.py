"""
config.py — SkyIQ Configuration
Change DB_TYPE here to switch between SQLite and MySQL.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── DATABASE TYPE ───────────────────────────────────────────
# Options: 'sqlite' or 'mysql'
DB_TYPE = 'mysql'   # ← change to 'mysql' when MySQL is ready

# ── MYSQL CONFIG (only used when DB_TYPE = 'mysql') ─────────
MYSQL_CONFIG = {
    'host':     os.getenv('DB_HOST'),
    'user':     os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
}

# ── FLASK CONFIG ────────────────────────────────────────────
SECRET_KEY = 'skyiq-secret-change-in-production'
DEBUG      = True
PORT       = 5000

AVIATIONSTACK_API_KEY = os.getenv('AVIATIONSTACK_API_KEY')
FLIGHTRADAR24_API_KEY = os.getenv('FLIGHTRADAR24_API_KEY')
