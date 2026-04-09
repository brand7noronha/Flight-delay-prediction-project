"""
config.py — SkyIQ Configuration
Change DB_TYPE here to switch between SQLite and MySQL.
"""

import os

# ── DATABASE TYPE ───────────────────────────────────────────
# Options: 'sqlite' or 'mysql'
DB_TYPE = 'mysql'   # ← change to 'mysql' when MySQL is ready

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

# ── API KEYS ─────────────────────────────────────────────────
# Set these values here, or export them as environment variables:
#   AVIATIONSTACK_API_KEY and FLIGHTRADAR24_API_KEY
AVIATIONSTACK_API_KEY = os.getenv('AVIATIONSTACK_API_KEY', '8bee62e0257c67a294c36e74a9846f41')
FLIGHTRADAR24_API_KEY = os.getenv('FLIGHTRADAR24_API_KEY', '019bfe6e-c828-705e-a195-153ef2b97f52|bExWsP0S0lX2FwWAmjUmx0RrWFXc7KukWe9cStIJ52c61fd4')
