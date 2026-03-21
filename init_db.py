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
    country      TEXT DEFAULT 'USA'
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

# ── SEED AIRPORTS ──────────────────────────────────────────
airports = [
    ('ATL','Hartsfield-Jackson Atlanta International','Atlanta','GA'),
    ('LAX','Los Angeles International','Los Angeles','CA'),
    ('ORD',"O'Hare International",'Chicago','IL'),
    ('DFW','Dallas/Fort Worth International','Dallas','TX'),
    ('DEN','Denver International','Denver','CO'),
    ('JFK','John F. Kennedy International','New York','NY'),
    ('SFO','San Francisco International','San Francisco','CA'),
    ('SEA','Seattle-Tacoma International','Seattle','WA'),
    ('LAS','Harry Reid International','Las Vegas','NV'),
    ('MCO','Orlando International','Orlando','FL'),
    ('MIA','Miami International','Miami','FL'),
    ('BOS','Logan International','Boston','MA'),
    ('MSP','Minneapolis-Saint Paul International','Minneapolis','MN'),
    ('PHX','Phoenix Sky Harbor International','Phoenix','AZ'),
    ('EWR','Newark Liberty International','Newark','NJ'),
    ('IAH','George Bush Intercontinental','Houston','TX'),
    ('CLT','Charlotte Douglas International','Charlotte','NC'),
    ('LGA','LaGuardia','New York','NY'),
    ('DTW','Detroit Metropolitan','Detroit','MI'),
    ('PHL','Philadelphia International','Philadelphia','PA'),
]
c.executemany(
    "INSERT OR IGNORE INTO airport (iata_code,airport_name,city,state) VALUES (?,?,?,?)",
    airports
)

conn.commit()
conn.close()
print("✅ Database created at:", DB_PATH)
print("✅ Tables created: airline, airport, route, weekly_flight_record,")
print("                   flight_aggregate, user, search_history,")
print("                   prediction_log, user_feedback")
print("✅ Seeded 15 airlines and 20 airports")
print("")
print("Now run:  python app.py")
