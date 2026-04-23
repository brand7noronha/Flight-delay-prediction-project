"""
init_db.py — Create and seed the SkyIQ SQLite database
Run once before starting the app:
    python init_db.py
"""
import sqlite3, os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR   = os.path.join(BASE_DIR, 'database')
DB_PATH  = os.path.join(DB_DIR, 'flights.db')

os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
c    = conn.cursor()

# ── CREATE TABLES ──────────────────────────────────────────
c.executescript("""

CREATE TABLE IF NOT EXISTS airline (
    airline_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    iata_code    TEXT UNIQUE NOT NULL,
    airline_name TEXT NOT NULL,
    country      TEXT DEFAULT 'USA'
);

CREATE TABLE IF NOT EXISTS airport (
    airport_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    iata_code    TEXT UNIQUE NOT NULL,
    airport_name TEXT NOT NULL,
    city         TEXT,
    state        TEXT,
    country      TEXT DEFAULT 'USA',
    latitude     REAL,
    longitude    REAL
);

CREATE TABLE IF NOT EXISTS route (
    route_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_airport_id INTEGER NOT NULL REFERENCES airport(airport_id),
    dest_airport_id   INTEGER NOT NULL REFERENCES airport(airport_id),
    distance_miles    REAL
);

CREATE TABLE IF NOT EXISTS weekly_flight_record (
    record_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number       TEXT NOT NULL,
    airline_id          INTEGER REFERENCES airline(airline_id),
    route_id            INTEGER REFERENCES route(route_id),
    flight_date         TEXT NOT NULL,
    scheduled_departure TEXT,
    scheduled_arrival   TEXT,
    departure_delay_min INTEGER DEFAULT 0,
    arrival_delay_min   INTEGER DEFAULT 0,
    is_delayed          INTEGER DEFAULT 0,
    is_cancelled        INTEGER DEFAULT 0,
    scheduled_time      INTEGER
);

CREATE TABLE IF NOT EXISTS flight_aggregate (
    aggregate_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number             TEXT NOT NULL,
    airline_id                INTEGER REFERENCES airline(airline_id),
    route_id                  INTEGER REFERENCES route(route_id),
    week_start_date           TEXT,
    week_end_date             TEXT,
    total_flights             INTEGER DEFAULT 0,
    total_delayed             INTEGER DEFAULT 0,
    total_on_time             INTEGER DEFAULT 0,
    total_cancelled           INTEGER DEFAULT 0,
    delay_rate                REAL DEFAULT 0,
    avg_delay_minutes         REAL DEFAULT 0,
    model_predicted_delay_rate REAL,
    prediction_accuracy       REAL
);

CREATE TABLE IF NOT EXISTS user (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS search_history (
    search_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES user(user_id),
    flight_number       TEXT,
    route_id            INTEGER REFERENCES route(route_id),
    search_date         TEXT,
    queried_flight_date TEXT,
    searched_at         TEXT
);

CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER REFERENCES user(user_id),
    route_id             INTEGER REFERENCES route(route_id),
    airline_id           INTEGER REFERENCES airline(airline_id),
    flight_number        TEXT,
    predicted_for_date   TEXT,
    delay_probability    REAL,
    risk_level           TEXT,
    predicted_delayed    INTEGER,
    actual_delayed       INTEGER,
    model_accuracy_at_time REAL,
    predicted_at         TEXT
);

CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id          INTEGER REFERENCES prediction_log(prediction_id),
    user_id                INTEGER REFERENCES user(user_id),
    user_confirmed_delayed INTEGER,
    comments               TEXT,
    submitted_at           TEXT
);
""")

# ── MIGRATE: add lat/lon columns to existing DBs ───────────
# Safe to run on an existing database — silently skips if already present.
for col_def in [
    "ALTER TABLE airport ADD COLUMN latitude  REAL",
    "ALTER TABLE airport ADD COLUMN longitude REAL",
]:
    try:
        c.execute(col_def)
    except Exception:
        pass  # column already exists

# ── SEED AIRLINES ──────────────────────────────────────────
airlines = [
    ('AA','American Airlines'),  ('DL','Delta Air Lines'),
    ('UA','United Airlines'),    ('WN','Southwest Airlines'),
    ('B6','JetBlue Airways'),    ('AS','Alaska Airlines'),
    ('NK','Spirit Airlines'),    ('F9','Frontier Airlines'),
    ('HA','Hawaiian Airlines'),  ('VX','Virgin America'),
    ('OO','SkyWest Airlines'),   ('MQ','Envoy Air'),
    ('EV','ExpressJet'),         ('YX','Republic Airways'),
    ('9E','Endeavor Air'),
]
c.executemany(
    "INSERT OR IGNORE INTO airline (iata_code, airline_name) VALUES (?,?)",
    airlines
)

# ── SEED AIRPORTS (with coordinates) ──────────────────────
# Columns: iata_code, airport_name, city, state, latitude, longitude
airports = [
    ('ATL','Hartsfield-Jackson Atlanta International','Atlanta',      'GA', 33.6407, -84.4277),
    ('LAX','Los Angeles International',               'Los Angeles',  'CA', 33.9416,-118.4085),
    ('ORD',"O'Hare International",                    'Chicago',      'IL', 41.9742, -87.9073),
    ('DFW','Dallas/Fort Worth International',         'Dallas',       'TX', 32.8998, -97.0403),
    ('DEN','Denver International',                    'Denver',       'CO', 39.8561,-104.6737),
    ('JFK','John F. Kennedy International',           'New York',     'NY', 40.6413, -73.7781),
    ('SFO','San Francisco International',             'San Francisco','CA', 37.6213,-122.3790),
    ('SEA','Seattle-Tacoma International',            'Seattle',      'WA', 47.4502,-122.3088),
    ('LAS','Harry Reid International',                'Las Vegas',    'NV', 36.0840,-115.1537),
    ('MCO','Orlando International',                   'Orlando',      'FL', 28.4312, -81.3081),
    ('MIA','Miami International',                     'Miami',        'FL', 25.7959, -80.2870),
    ('BOS','Logan International',                     'Boston',       'MA', 42.3656, -71.0096),
    ('MSP','Minneapolis-Saint Paul International',    'Minneapolis',  'MN', 44.8848, -93.2223),
    ('PHX','Phoenix Sky Harbor International',        'Phoenix',      'AZ', 33.4353,-112.0058),
    ('EWR','Newark Liberty International',            'Newark',       'NJ', 40.6895, -74.1745),
    ('IAH','George Bush Intercontinental',            'Houston',      'TX', 29.9902, -95.3368),
    ('CLT','Charlotte Douglas International',         'Charlotte',    'NC', 35.2144, -80.9473),
    ('LGA','LaGuardia',                               'New York',     'NY', 40.7769, -73.8740),
    ('DTW','Detroit Metropolitan',                    'Detroit',      'MI', 42.2162, -83.3554),
    ('PHL','Philadelphia International',              'Philadelphia', 'PA', 39.8721, -75.2437),
]
c.executemany(
    """INSERT OR IGNORE INTO airport
       (iata_code, airport_name, city, state, latitude, longitude)
       VALUES (?,?,?,?,?,?)""",
    airports
)

# Update coordinates for airports already seeded without them
c.executemany(
    """UPDATE airport SET latitude=?, longitude=?
       WHERE iata_code=? AND (latitude IS NULL OR longitude IS NULL)""",
    [(lat, lon, code) for code, *_, lat, lon in airports]
)

conn.commit()
conn.close()
print("✅ Database created at:", DB_PATH)
print("✅ Tables created: airline, airport (with lat/lon), route,")
print("                   weekly_flight_record, flight_aggregate, user,")
print("                   search_history, prediction_log, user_feedback")
print("✅ Seeded 15 airlines and 20 airports (with coordinates)")
print("")
print("Now run:  python app.py")
