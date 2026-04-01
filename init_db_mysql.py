"""
init_db_mysql.py — Create and seed MySQL database for SkyIQ
Run ONCE after MySQL server is running:
    python init_db_mysql.py
"""

import pymysql
from config import MYSQL_CONFIG

print("Connecting to MySQL...")
conn = pymysql.connect(
    host     = MYSQL_CONFIG['host'],
    user     = MYSQL_CONFIG['user'],
    password = MYSQL_CONFIG['password'],
    database = MYSQL_CONFIG['database'],
    charset  = 'utf8mb4'
)
c = conn.cursor()
print("✅ Connected to MySQL")

# ── CREATE TABLES ──────────────────────────────────────────
tables = [

"""
CREATE TABLE IF NOT EXISTS airline (
    airline_id   INT AUTO_INCREMENT PRIMARY KEY,
    iata_code    VARCHAR(5)  UNIQUE NOT NULL,
    airline_name VARCHAR(100) NOT NULL,
    country      VARCHAR(50)  DEFAULT 'USA'
)
""",

"""
CREATE TABLE IF NOT EXISTS airport (
    airport_id   INT AUTO_INCREMENT PRIMARY KEY,
    iata_code    VARCHAR(5)  UNIQUE NOT NULL,
    airport_name VARCHAR(150) NOT NULL,
    city         VARCHAR(100),
    state        VARCHAR(50),
    country      VARCHAR(50) DEFAULT 'USA'
)
""",

"""
CREATE TABLE IF NOT EXISTS route (
    route_id          INT AUTO_INCREMENT PRIMARY KEY,
    origin_airport_id INT NOT NULL,
    dest_airport_id   INT NOT NULL,
    distance_miles    FLOAT,
    FOREIGN KEY (origin_airport_id) REFERENCES airport(airport_id),
    FOREIGN KEY (dest_airport_id)   REFERENCES airport(airport_id)
)
""",

"""
CREATE TABLE IF NOT EXISTS weekly_flight_record (
    record_id           INT AUTO_INCREMENT PRIMARY KEY,
    flight_number       VARCHAR(10) NOT NULL,
    airline_id          INT,
    route_id            INT,
    flight_date         DATE NOT NULL,
    scheduled_departure TIME,
    scheduled_arrival   TIME,
    departure_delay_min INT     DEFAULT 0,
    arrival_delay_min   INT     DEFAULT 0,
    is_delayed          TINYINT DEFAULT 0,
    is_cancelled        TINYINT DEFAULT 0,
    scheduled_time      INT,
    FOREIGN KEY (airline_id) REFERENCES airline(airline_id),
    FOREIGN KEY (route_id)   REFERENCES route(route_id)
)
""",

"""
CREATE TABLE IF NOT EXISTS flight_aggregate (
    aggregate_id               INT AUTO_INCREMENT PRIMARY KEY,
    flight_number              VARCHAR(10) NOT NULL,
    airline_id                 INT,
    route_id                   INT,
    week_start_date            DATE,
    week_end_date              DATE,
    total_flights              INT   DEFAULT 0,
    total_delayed              INT   DEFAULT 0,
    total_on_time              INT   DEFAULT 0,
    total_cancelled            INT   DEFAULT 0,
    delay_rate                 FLOAT DEFAULT 0,
    avg_delay_minutes          FLOAT DEFAULT 0,
    model_predicted_delay_rate FLOAT,
    prediction_accuracy        FLOAT,
    FOREIGN KEY (airline_id) REFERENCES airline(airline_id),
    FOREIGN KEY (route_id)   REFERENCES route(route_id)
)
""",

"""
CREATE TABLE IF NOT EXISTS user (
    user_id       INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(80)  NOT NULL,
    email         VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(200) NOT NULL,
    created_at    DATETIME,
    last_login    DATETIME
)
""",

"""
CREATE TABLE IF NOT EXISTS search_history (
    search_id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id             INT,
    flight_number       VARCHAR(10),
    route_id            INT,
    search_date         DATE,
    queried_flight_date DATE,
    searched_at         DATETIME,
    FOREIGN KEY (user_id)   REFERENCES user(user_id),
    FOREIGN KEY (route_id)  REFERENCES route(route_id)
)
""",

"""
CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id                INT,
    route_id               INT,
    airline_id             INT,
    flight_number          VARCHAR(10),
    predicted_for_date     DATE,
    delay_probability      FLOAT,
    risk_level             VARCHAR(10),
    predicted_delayed      TINYINT,
    actual_delayed         TINYINT,
    model_accuracy_at_time FLOAT,
    predicted_at           DATETIME,
    FOREIGN KEY (user_id)   REFERENCES user(user_id),
    FOREIGN KEY (route_id)  REFERENCES route(route_id),
    FOREIGN KEY (airline_id) REFERENCES airline(airline_id)
)
""",

"""
CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id            INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id          INT,
    user_id                INT,
    user_confirmed_delayed TINYINT,
    comments               TEXT,
    submitted_at           DATETIME,
    FOREIGN KEY (prediction_id) REFERENCES prediction_log(prediction_id),
    FOREIGN KEY (user_id)       REFERENCES user(user_id)
)
"""
]

for sql in tables:
    c.execute(sql)
    print(f"  ✅ Table created")

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
    "INSERT IGNORE INTO airline (iata_code, airline_name) VALUES (%s, %s)",
    airlines
)
print(f"✅ Seeded {len(airlines)} airlines")

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
    "INSERT IGNORE INTO airport (iata_code,airport_name,city,state) VALUES (%s,%s,%s,%s)",
    airports
)
print(f"✅ Seeded {len(airports)} airports")

conn.commit()
conn.close()

print("")
print("═══════════════════════════════════════")
print("✅ MySQL database setup complete!")
print("═══════════════════════════════════════")
print("Tables created:")
print("  airline, airport, route")
print("  weekly_flight_record, flight_aggregate")
print("  user, search_history")
print("  prediction_log, user_feedback")
print("")
print("Next step:")
print("  1. Open config.py")
print("  2. Change DB_TYPE = 'sqlite'  →  DB_TYPE = 'mysql'")
print("  3. Run: python app.py")
