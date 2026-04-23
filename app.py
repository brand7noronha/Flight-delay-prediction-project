"""
app.py — SkyIQ Flask Backend
Flight Delay Prediction System
MCA Group 3 — Apa Mestry (2512) & Brandon Noronha (2514)

Install:
    pip install flask pymysql joblib scikit-learn lightgbm pandas numpy
    python init_db_mysql.py
    python app.py
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
import hashlib, logging, os, random, math, re
from datetime import datetime, date

import requests

logger = logging.getLogger(__name__)

# ── CONFIG & DB ────────────────────────────────────────────
from config import (
    SECRET_KEY, DEBUG, PORT,
    AVIATIONSTACK_API_KEY, FLIGHTRADAR24_API_KEY,
)
from db import get_db, query, query_one, execute, commit, close

app = Flask(__name__)
app.secret_key = SECRET_KEY


# ══════════════════════════════════════════════════════════
# ML MODEL LOADING
# ══════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

ML_READY   = False   # flips to True if the .pkl file loads successfully
model      = None
label_encoders = {}  # built in-code; no separate encoder .pkl needed
MODEL_NAME = 'Mock'

# ── Categorical vocabulary (must match training data) ──────
_AIRLINE_CODES = sorted([
    '9E','AA','AS','B6','DL','EV','F9','HA',
    'MQ','NK','OO','UA','VX','WN','YX'
])
_AIRPORT_CODES = sorted(set([
    'ATL','BOS','CLT','DEN','DFW','DTW','EWR',
    'IAH','JFK','LAS','LAX','LGA','MCO','MIA',
    'MSP','ORD','PHX','PHL','SEA','SFO'
]))
_TIME_PERIODS  = sorted(['afternoon','evening','late_night','morning'])


def _build_label_encoders():
    try:
        from sklearn.preprocessing import LabelEncoder
        for col, vocab in [
            ('AIRLINE',              _AIRLINE_CODES),
            ('ORIGIN_AIRPORT',       _AIRPORT_CODES),
            ('DESTINATION_AIRPORT',  _AIRPORT_CODES),
            ('TIME_PERIOD',          _TIME_PERIODS),
        ]:
            le = LabelEncoder()
            le.fit(vocab)
            label_encoders[col] = le
        return True
    except ImportError:
        return False


try:
    import joblib
    import pandas as pd

    _candidates = [
        os.path.join(BASE_DIR, 'model', 'flight_delay_regressor.pkl'),
        os.path.join(BASE_DIR, 'flight_delay_regressor.pkl'),
    ]
    _model_path = next((p for p in _candidates if os.path.exists(p)), None)

    if _model_path is None:
        raise FileNotFoundError(
            "flight_delay_regressor.pkl not found in model/ or current directory"
        )

    model      = joblib.load(_model_path)
    MODEL_NAME = f"{type(model).__name__} (LightGBM)"

    if _build_label_encoders():
        ML_READY = True
        print(f"✅ ML model loaded: {MODEL_NAME}")
        print(f"   Path: {_model_path}")
    else:
        print("⚠️  scikit-learn not installed — falling back to Mock mode")

except FileNotFoundError as e:
    print(f"⚠️  {e}")
    print("   Place flight_delay_regressor.pkl in model/ to enable real predictions")
except ImportError as e:
    print(f"⚠️  Missing package: {e} — running in Mock mode")
except Exception as e:
    print(f"⚠️  Model load error: {e} — running in Mock mode")


def _encode(col: str, value: str) -> int:
    le = label_encoders.get(col)
    if le is None:
        return 0
    try:
        return int(le.transform([str(value)])[0])
    except ValueError:
        return 0


def _minutes_to_prob(predicted_minutes: float) -> float:
    return round(1.0 / (1.0 + math.exp(-(predicted_minutes - 15.0) / 20.0)), 4)


# ══════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════
def predict_delay(airline, origin, destination, month,
                  day_of_week, scheduled_departure_hhmm,
                  departure_delay, distance, scheduled_time):
    dep_hour = int(str(scheduled_departure_hhmm).zfill(4)[:2]) \
               if scheduled_departure_hhmm else 12

    if   dep_hour < 6:  time_period = 'late_night'
    elif dep_hour < 12: time_period = 'morning'
    elif dep_hour < 18: time_period = 'afternoon'
    else:               time_period = 'evening'

    if ML_READY:
        try:
            row = pd.DataFrame([{
                'AIRLINE':             _encode('AIRLINE',             str(airline)),
                'ORIGIN_AIRPORT':      _encode('ORIGIN_AIRPORT',      str(origin)),
                'DESTINATION_AIRPORT': _encode('DESTINATION_AIRPORT', str(destination)),
                'MONTH':               int(month),
                'DAY_OF_WEEK':         int(day_of_week),
                'DEPARTURE_HOUR':      int(dep_hour),
                'DEPARTURE_DELAY':     float(departure_delay),
                'DISTANCE':            float(distance),
                'SCHEDULED_TIME':      float(scheduled_time),
                'TIME_PERIOD':         _encode('TIME_PERIOD',         time_period),
            }])

            predicted_minutes = float(model.predict(row)[0])
            prob = _minutes_to_prob(predicted_minutes)
            pred = prob >= 0.50
            risk = ('Low'    if prob < 0.35 else
                    'Medium' if prob < 0.65 else 'High')

            return {
                'is_delayed':           pred,
                'delay_probability':    prob,
                'predicted_delay_min':  round(predicted_minutes, 1),
                'risk_level':           risk,
                'departure_hour':       dep_hour,
                'time_period':          time_period,
                'model_name':           MODEL_NAME,
                'message': (
                    f"{'Likely DELAYED' if pred else 'Likely ON TIME'} "
                    f"— {round(prob * 100, 1)}% delay probability "
                    f"({risk} Risk, ~{round(predicted_minutes)} min predicted)"
                )
            }

        except Exception as e:
            print(f"⚠️  Prediction error: {e} — falling back to mock")

    # ── MOCK FALLBACK ──────────────────────────────────────
    base_prob = 0.28
    if dep_hour >= 18:              base_prob += 0.20
    if dep_hour >= 21:              base_prob += 0.10
    if float(departure_delay) > 0: base_prob += 0.25
    if month in [7, 12, 1]:        base_prob += 0.10
    if day_of_week in [5, 1]:      base_prob += 0.05

    prob = round(min(max(base_prob + random.uniform(-0.05, 0.05), 0.05), 0.95), 4)
    pred = prob >= 0.5
    risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.65 else 'High')

    return {
        'is_delayed':           pred,
        'delay_probability':    prob,
        'predicted_delay_min':  None,
        'risk_level':           risk,
        'departure_hour':       dep_hour,
        'time_period':          time_period,
        'model_name':           'Mock (flight_delay_regressor.pkl not found)',
        'message': (
            f"{'Likely DELAYED' if pred else 'Likely ON TIME'} "
            f"— {round(prob * 100, 1)}% delay probability ({risk} Risk)"
        )
    }


# ── API SUPPORT ──────────────────────────────────────────────
API_CALL_COUNTER = 0
API_LAST_CALL = None

# ── AIRPORT COORDINATES ─────────────────────────────────────
# These are used by _distance_between_airports() as a fallback when the
# airport table does not yet have latitude/longitude populated.
# Primary source: the airport table (queried live and cached in _coord_cache).
_AIRPORT_COORDINATES_FALLBACK = {
    'ATL': (33.6407, -84.4277),  'BOS': (42.3656, -71.0096),
    'CLT': (35.2144, -80.9473),  'DEN': (39.8561,-104.6737),
    'DFW': (32.8998, -97.0403),  'DTW': (42.2162, -83.3554),
    'EWR': (40.6895, -74.1745),  'IAH': (29.9902, -95.3368),
    'JFK': (40.6413, -73.7781),  'LAS': (36.0840,-115.1537),
    'LAX': (33.9416,-118.4085),  'LGA': (40.7769, -73.8740),
    'MCO': (28.4312, -81.3081),  'MIA': (25.7959, -80.2870),
    'MSP': (44.8848, -93.2223),  'ORD': (41.9742, -87.9073),
    'PHX': (33.4353,-112.0058),  'PHL': (39.8721, -75.2437),
    'SEA': (47.4502,-122.3088),  'SFO': (37.6213,-122.3790),
}

# Simple in-process cache so we don't hit the DB on every haversine call.
_coord_cache: dict = {}


def _get_airport_coords(iata_code: str):
    """
    Return (latitude, longitude) for an IATA code.
    Tries the DB first (caching the result), then falls back to the
    hardcoded dict so the app works even before init_db has run.
    """
    if not iata_code:
        return None
    iata_code = iata_code.upper()

    if iata_code in _coord_cache:
        return _coord_cache[iata_code]

    try:
        conn = get_db()
        row = query_one(
            conn,
            "SELECT latitude, longitude FROM airport WHERE iata_code = ?",
            (iata_code,)
        )
        close(conn)
        if row and row.get('latitude') is not None and row.get('longitude') is not None:
            coords = (float(row['latitude']), float(row['longitude']))
            _coord_cache[iata_code] = coords
            return coords
    except Exception:
        pass  # DB not ready yet; fall through to the hardcoded dict

    coords = _AIRPORT_COORDINATES_FALLBACK.get(iata_code)
    if coords:
        _coord_cache[iata_code] = coords
    return coords


def increment_api_counter(provider, endpoint, success=True, note=None):
    global API_CALL_COUNTER, API_LAST_CALL
    API_CALL_COUNTER += 1
    API_LAST_CALL = {
        'provider': provider,
        'endpoint': endpoint,
        'success': success,
        'note': note,
        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
    }


def _parse_iso_datetime(value):
    if not value:
        return None
    try:
        if isinstance(value, str) and value.endswith('Z'):
            value = value[:-1] + '+00:00'
        return datetime.fromisoformat(value)
    except Exception:
        try:
            return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except Exception:
            return None


def _get_hhmm_from_datetime(value):
    if not value:
        return None
    try:
        return int(value.strftime('%H%M'))
    except Exception:
        return None


def _distance_between_airports(origin, destination):
    """Haversine distance in miles between two IATA airport codes."""
    if not origin or not destination:
        return None
    coords_a = _get_airport_coords(origin)
    coords_b = _get_airport_coords(destination)
    if not coords_a or not coords_b:
        return None

    lat1, lon1 = coords_a
    lat2, lon2 = coords_b
    radius = 3959.0  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(round(radius * c))


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def lookup_with_aviationstack(flight_no, flight_date=None):
    if not AVIATIONSTACK_API_KEY:
        return None

    url = 'https://api.aviationstack.com/v1/flights'
    params = {'access_key': AVIATIONSTACK_API_KEY, 'flight_iata': flight_no, 'limit': 10}
    if flight_date:
        params['flight_date'] = flight_date

    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        increment_api_counter('aviationstack', url, success=False, note=str(exc))
        return None

    flights = data.get('data') or []
    if not flights:
        increment_api_counter('aviationstack', url, success=False, note='no flights returned')
        return None

    flight = flights[0]
    departure = flight.get('departure', {}) or {}
    arrival = flight.get('arrival', {}) or {}

    sched_dep_dt  = _parse_iso_datetime(departure.get('scheduled'))
    actual_dep_dt = _parse_iso_datetime(departure.get('actual'))
    sched_arr_dt  = _parse_iso_datetime(arrival.get('scheduled'))

    scheduled_departure = _get_hhmm_from_datetime(sched_dep_dt)
    scheduled_time = None
    if sched_dep_dt and sched_arr_dt:
        duration = int(round((sched_arr_dt - sched_dep_dt).total_seconds() / 60.0))
        scheduled_time = abs(duration) if duration > 0 else None

    departure_delay = None
    if departure.get('delay') is not None:
        departure_delay = _safe_int(departure.get('delay'), None)
    elif actual_dep_dt and sched_dep_dt:
        departure_delay = _safe_int((actual_dep_dt - sched_dep_dt).total_seconds() / 60.0, None)

    flight_date_value = None
    if sched_dep_dt:
        flight_date_value = sched_dep_dt.strftime('%Y-%m-%d')
    elif flight_date:
        flight_date_value = flight_date

    details = {
        'source': 'Aviationstack',
        'airline': flight.get('airline', {}).get('iata'),
        'origin': departure.get('iata'),
        'destination': arrival.get('iata'),
        'flight_date': flight_date_value,
        'scheduled_departure': f"{scheduled_departure:04d}" if scheduled_departure is not None else None,
        'scheduled_time': scheduled_time,
        'departure_delay': departure_delay,
        'distance': _distance_between_airports(departure.get('iata'), arrival.get('iata')),
        'message': 'Data populated from Aviationstack API',
    }
    increment_api_counter('aviationstack', url, success=True, note=f"flight_iata={flight_no}")
    return details


def lookup_with_flightradar24(flight_no, flight_date=None):
    if not FLIGHTRADAR24_API_KEY:
        return None

    url = 'https://fr24api.flightradar24.com/api/live/flight-positions/light'
    params = {'flight': flight_no}
    headers = {
        'Accept': 'application/json',
        'Accept-Version': 'v1',
        'Authorization': f'Bearer {FLIGHTRADAR24_API_KEY}'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=12)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        increment_api_counter('flightradar24', url, success=False, note=str(exc))
        return None

    results = data.get('data') or []
    if not results:
        increment_api_counter('flightradar24', url, success=False, note='no flights returned')
        return None

    item = results[0]
    flight = item.get('flight') or item
    departure = item.get('airport', {}).get('origin', {}) or {}
    arrival = item.get('airport', {}).get('destination', {}) or {}

    sched_dep_dt = _parse_iso_datetime(departure.get('scheduled'))
    sched_arr_dt = _parse_iso_datetime(arrival.get('scheduled'))
    scheduled_departure = _get_hhmm_from_datetime(sched_dep_dt)
    scheduled_time = None
    if sched_dep_dt and sched_arr_dt:
        duration = int(round((sched_arr_dt - sched_dep_dt).total_seconds() / 60.0))
        scheduled_time = abs(duration) if duration > 0 else None

    departure_delay = None
    if departure.get('delay') is not None:
        departure_delay = _safe_int(departure.get('delay'), None)

    flight_date_value = None
    if sched_dep_dt:
        flight_date_value = sched_dep_dt.strftime('%Y-%m-%d')
    elif flight_date:
        flight_date_value = flight_date

    details = {
        'source': 'Flightradar24',
        'airline': (flight.get('airline', {}) or {}).get('iata') or flight.get('airline_iata'),
        'origin': departure.get('iata'),
        'destination': arrival.get('iata'),
        'flight_date': flight_date_value,
        'scheduled_departure': f"{scheduled_departure:04d}" if scheduled_departure is not None else None,
        'scheduled_time': scheduled_time,
        'departure_delay': departure_delay,
        'distance': _distance_between_airports(departure.get('iata'), arrival.get('iata')),
        'message': 'Data populated from Flightradar24 API',
    }
    increment_api_counter('flightradar24', url, success=True, note=f"flight={flight_no}")
    return details


def lookup_flight_details(flight_no, flight_date=None):
    if not flight_no:
        return {}
    lookup = lookup_with_aviationstack(flight_no, flight_date)
    if lookup:
        return lookup
    return lookup_with_flightradar24(flight_no, flight_date)


# ── PASSWORD HELPERS ────────────────────────────────────────
def hash_password(password):
    salt   = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def check_password(password, stored):
    try:
        salt, hashed = stored.split(':', 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


# ── VALIDATION HELPERS ──────────────────────────────────────
_EMAIL_RE    = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
_USERNAME_RE = re.compile(r'^[a-zA-Z0-9_]+$')

def validate_email(email: str) -> tuple[bool, str | None]:
    if not email:
        return False, "Email is required."
    if not _EMAIL_RE.match(email):
        return False, "Enter a valid email address."
    return True, None

def validate_password(password: str) -> tuple[bool, str | None]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    return True, None

def validate_username(username: str) -> tuple[bool, str | None]:
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 30:
        return False, "Username must be 30 characters or fewer."
    if not _USERNAME_RE.match(username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, None


# ── DATE HELPER ─────────────────────────────────────────────
def format_date(value):
    if not value:
        return 'N/A'
    if isinstance(value, str):
        try:
            value = datetime.strptime(value[:19], '%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                value = datetime.strptime(value[:10], '%Y-%m-%d')
            except Exception:
                return str(value)[:7]
    return value.strftime('%b %Y')

app.jinja_env.filters['fmtdate'] = format_date


# ── MOCK FALLBACK DATA ──────────────────────────────────────
MOCK_AIRLINES = [
    ('AA', 'American Airlines'),  ('DL', 'Delta Air Lines'),
    ('UA', 'United Airlines'),    ('WN', 'Southwest Airlines'),
    ('B6', 'JetBlue Airways'),    ('AS', 'Alaska Airlines'),
    ('NK', 'Spirit Airlines'),    ('F9', 'Frontier Airlines'),
    ('HA', 'Hawaiian Airlines'),  ('VX', 'Virgin America'),
]

MOCK_AIRPORTS = [
    ('ATL', 'Hartsfield-Jackson Atlanta',      'Atlanta'),
    ('LAX', 'Los Angeles International',       'Los Angeles'),
    ('ORD', "O'Hare International",            'Chicago'),
    ('DFW', 'Dallas/Fort Worth International', 'Dallas'),
    ('DEN', 'Denver International',            'Denver'),
    ('JFK', 'John F. Kennedy International',   'New York'),
    ('SFO', 'San Francisco International',     'San Francisco'),
    ('SEA', 'Seattle-Tacoma International',    'Seattle'),
    ('LAS', 'Harry Reid International',        'Las Vegas'),
    ('MCO', 'Orlando International',           'Orlando'),
    ('MIA', 'Miami International',             'Miami'),
    ('BOS', 'Logan International',             'Boston'),
    ('MSP', 'Minneapolis-Saint Paul',          'Minneapolis'),
    ('PHX', 'Phoenix Sky Harbor',              'Phoenix'),
    ('EWR', 'Newark Liberty International',    'Newark'),
]

MOCK_TOP_ROUTES = [
    {'origin': 'LAX', 'destination': 'JFK', 'delay_rate': 68},
    {'origin': 'ORD', 'destination': 'LGA', 'delay_rate': 62},
    {'origin': 'SFO', 'destination': 'LAX', 'delay_rate': 57},
    {'origin': 'ATL', 'destination': 'ORD', 'delay_rate': 54},
    {'origin': 'DFW', 'destination': 'LAX', 'delay_rate': 51},
    {'origin': 'JFK', 'destination': 'BOS', 'delay_rate': 49},
    {'origin': 'MIA', 'destination': 'JFK', 'delay_rate': 47},
    {'origin': 'DEN', 'destination': 'ORD', 'delay_rate': 44},
    {'origin': 'SEA', 'destination': 'SFO', 'delay_rate': 42},
    {'origin': 'LAS', 'destination': 'LAX', 'delay_rate': 38},
]


# ── LOOKUP HELPERS ──────────────────────────────────────────
def get_airlines():
    try:
        conn = get_db()
        rows = query(conn, "SELECT iata_code, airline_name FROM airline ORDER BY airline_name")
        close(conn)
        if rows:
            return [(r['iata_code'], r['airline_name']) for r in rows]
    except Exception:
        pass
    return MOCK_AIRLINES

def get_airports():
    try:
        conn = get_db()
        rows = query(conn, "SELECT iata_code, airport_name, city FROM airport ORDER BY city")
        close(conn)
        if rows:
            return [(r['iata_code'], r['airport_name'], r['city']) for r in rows]
    except Exception:
        pass
    return MOCK_AIRPORTS


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.context_processor
def inject_api_debug():
    return {
        'api_debug': {
            'call_count': API_CALL_COUNTER,
            'last_call': API_LAST_CALL,
        }
    }


@app.route('/lookup-flight')
def lookup_flight():
    flight_no   = request.args.get('flight_no', '').strip().upper()
    flight_date = request.args.get('flight_date')
    if not flight_no:
        return jsonify({'ok': False, 'message': 'flight_no query parameter is required'}), 400

    lookup = lookup_flight_details(flight_no, flight_date)
    if not lookup or not lookup.get('source'):
        return jsonify({'ok': False, 'message': 'Unable to find flight details for ' + flight_no}), 404

    return jsonify({'ok': True, 'lookup': lookup})


# ── HOME ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── PREDICT ─────────────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    airlines = get_airlines()
    airports = get_airports()
    prefill  = request.args.to_dict()

    if request.method == 'POST':
        form        = request.form
        flight_no   = form.get('flight_no', '').strip().upper()
        airline     = form.get('airline', '').strip().upper()
        origin      = form.get('origin', '').strip().upper()
        destination = form.get('destination', '').strip().upper()
        flight_date = form.get('flight_date', '').strip() or None
        sched_dep   = form.get('scheduled_departure', '').replace(':', '').strip() or None
        distance    = form.get('distance', '').strip() or None
        dep_delay   = form.get('departure_delay', '').strip() or None
        sched_time  = form.get('scheduled_time', '').strip() or None

        if flight_no and (not airline or not origin or not destination
                          or not flight_date or not sched_dep
                          or not distance or not sched_time):
            lookup = lookup_flight_details(flight_no, flight_date)
            if lookup:
                airline     = airline or (lookup.get('airline') or airline)
                origin      = origin or (lookup.get('origin') or origin)
                destination = destination or (lookup.get('destination') or destination)
                flight_date = flight_date or lookup.get('flight_date')
                sched_dep   = sched_dep or lookup.get('scheduled_departure')
                distance    = distance or lookup.get('distance')
                dep_delay   = dep_delay or lookup.get('departure_delay')
                sched_time  = sched_time or lookup.get('scheduled_time')
                api_lookup  = lookup
            else:
                api_lookup = {'source': None, 'message': 'Lookup failed or no external data found.'}
        else:
            api_lookup = None

        if not flight_date:
            flight_date = str(date.today())

        try:
            dt    = datetime.strptime(flight_date, '%Y-%m-%d')
            month = dt.month
            dow   = dt.isoweekday()
        except Exception:
            month, dow = 6, 3

        sched_dep  = sched_dep or '1200'
        distance   = distance or 500
        dep_delay  = dep_delay or 0
        sched_time = sched_time or 120

        prediction = predict_delay(
            airline=airline,
            origin=origin,
            destination=destination,
            month=month,
            day_of_week=dow,
            scheduled_departure_hhmm=int(sched_dep) if sched_dep else 1200,
            departure_delay=float(dep_delay),
            distance=float(distance),
            scheduled_time=float(sched_time)
        )

        formatted_sched_dep = sched_dep
        if formatted_sched_dep and len(str(formatted_sched_dep)) == 4:
            formatted_sched_dep = f"{str(formatted_sched_dep)[:2]}:{str(formatted_sched_dep)[2:]}"

        prediction.update({
            'flight_number':       flight_no,
            'airline':             airline,
            'origin':              origin,
            'destination':         destination,
            'flight_date':         flight_date,
            'scheduled_departure': formatted_sched_dep or '12:00',
            'distance':            distance,
            'departure_delay':     dep_delay,
            'lookup':              api_lookup,
        })

        history_data = None
        try:
            conn = get_db()
            hist = query_one(conn, """
                SELECT total_flights, total_on_time, total_delayed,
                       delay_rate, avg_delay_minutes
                FROM   flight_aggregate
                WHERE  flight_number = ?
                ORDER  BY week_start_date DESC
                LIMIT  1
            """, (flight_no,))
            close(conn)
            if hist:
                history_data = dict(hist)
        except Exception:
            pass

        if not history_data:
            history_data = {
                'total_flights':     random.randint(5, 14),
                'total_on_time':     random.randint(3, 9),
                'total_delayed':     random.randint(1, 5),
                'delay_rate':        round(random.uniform(0.2, 0.7), 2),
                'avg_delay_minutes': random.randint(18, 55),
            }
        prediction['history'] = history_data

        prediction_id = None
        if session.get('user_id'):
            conn = None
            try:
                conn = get_db()
                uid = int(session['user_id'])
                prediction_id = execute(conn, """
                    INSERT INTO prediction_log
                      (user_id, flight_number, predicted_for_date,
                       delay_probability, risk_level,
                       predicted_delayed, predicted_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (uid, flight_no, flight_date,
                      prediction['delay_probability'],
                      prediction['risk_level'],
                      int(prediction['is_delayed']),
                      str(datetime.utcnow())))
                execute(conn, """
                    INSERT INTO search_history
                      (user_id, flight_number, search_date,
                       queried_flight_date, searched_at)
                    VALUES (?,?,?,?,?)
                """, (uid, flight_no,
                      str(date.today()), flight_date,
                      str(datetime.utcnow())))
                commit(conn)
            except Exception:
                logger.exception("Failed to persist prediction for user_id=%s", session.get('user_id'))
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
            finally:
                if conn is not None:
                    close(conn)

        prediction['prediction_id'] = prediction_id
        return render_template('result.html', result=prediction)

    return render_template('predict.html',
                           airlines=airlines,
                           airports=airports,
                           prefill=prefill)


# # ── DASHBOARD ───────────────────────────────────────────────
# @app.route('/dashboard')
# def dashboard():
#     stats = {
#         'total_flights':  1284,
#         'delay_rate':     38.4,
#         'avg_delay':      34,
#         'model_accuracy': 78,
#     }
#     chart_data = {
#         'airline': {
#             'labels': ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'VX'],
#             'values': [42, 35, 48, 29, 38, 31, 55, 52, 22, 36],
#         },
#         'hour': {
#             'labels': [f"{h}:00" for h in range(0, 24)],
#             'values': [18, 15, 12, 10, 11, 14, 19, 25, 28, 30,
#                        32, 34, 35, 36, 38, 40, 42, 45, 48, 50,
#                        47, 44, 38, 28],
#         },
#         'month': {
#             'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
#             'values': [45, 38, 35, 32, 36, 40, 52, 48, 34, 30, 35, 50],
#         },
#         'day': {
#             'labels': ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
#             'values': [35, 42, 34, 33, 38, 48, 30],
#         },
#     }
#     return render_template('dashboard.html',
#                            stats=stats,
#                            chart_data=chart_data,
#                            top_routes=MOCK_TOP_ROUTES,
#                            airlines=get_airlines())


@app.route('/dashboard')
def dashboard():
    """Dynamic dashboard with real database data"""
    
    # Get filter parameters
    airline_filter = request.args.get('airline', 'ALL')
    month_filter = request.args.get('month', '0')
    
    print(f"Dashboard filters - Airline: {airline_filter}, Month: {month_filter}")  # Debug
    
    try:
        conn = get_db()
        
        # Build WHERE clauses
        where_parts = []
        params = []
        
        if airline_filter != 'ALL':
            where_parts.append("a.iata_code = %s")
            params.append(airline_filter)
        
        if month_filter != '0':
            where_parts.append("MONTH(wfr.flight_date) = %s")
            params.append(month_filter)
        
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        
        # ── STATS (Filtered) ─────────────────────────────
        stats_query = f"""
            SELECT 
                COUNT(*) as total_flights,
                COALESCE(AVG(wfr.is_delayed), 0) * 100 as delay_rate,
                COALESCE(AVG(CASE WHEN wfr.arrival_delay_min > 0 THEN wfr.arrival_delay_min ELSE NULL END), 0) as avg_delay
            FROM weekly_flight_record wfr
            JOIN airline a ON wfr.airline_id = a.airline_id
            WHERE {where_clause}
        """
        
        stats_row = query_one(conn, stats_query, params if params else None)
        
        # Model accuracy (from user feedback)
        accuracy_row = query_one(conn, """
            SELECT 
                COUNT(*) as total_feedback,
                COALESCE(AVG(CASE 
                    WHEN predicted_delayed = actual_delayed THEN 1 
                    ELSE 0 
                END), 0.78) * 100 as accuracy
            FROM prediction_log
            WHERE actual_delayed IS NOT NULL
        """)
        
        stats = {
            'total_flights': int(stats_row['total_flights']) if stats_row and stats_row['total_flights'] else 0,
            'delay_rate': round(stats_row['delay_rate'], 1) if stats_row else 0,
            'avg_delay': int(stats_row['avg_delay']) if stats_row else 0,
            'model_accuracy': round(accuracy_row['accuracy'], 0) if accuracy_row else 78,
        }
        
        # ── CHART 1: Delay by Airline (Filtered by month) ──
        chart_where = "1=1"
        if month_filter != '0':
            chart_where = f"MONTH(wfr.flight_date) = {month_filter}"
        
        airline_query = f"""
            SELECT 
                a.iata_code as airline,
                COALESCE(AVG(wfr.is_delayed), 0) * 100 as delay_rate
            FROM weekly_flight_record wfr
            JOIN airline a ON wfr.airline_id = a.airline_id
            WHERE {chart_where}
            GROUP BY a.iata_code
            ORDER BY delay_rate DESC
            LIMIT 10
        """
        
        airline_rows = query(conn, airline_query)
        chart_airline = {
            'labels': [row['airline'] for row in airline_rows],
            'values': [round(row['delay_rate'], 1) for row in airline_rows]
        } if airline_rows else {'labels': [], 'values': []}
        
        # ── CHART 2: Delay by Hour (Filtered) ────────────
        hour_where = "1=1"
        hour_params = []
        
        if airline_filter != 'ALL':
            hour_where = "a.iata_code = %s"
            hour_params.append(airline_filter)
        
        hour_query = f"""
            SELECT 
                HOUR(wfr.scheduled_departure) as hour,
                COALESCE(AVG(wfr.is_delayed), 0) * 100 as delay_rate
            FROM weekly_flight_record wfr
            JOIN airline a ON wfr.airline_id = a.airline_id
            WHERE {hour_where}
            GROUP BY HOUR(wfr.scheduled_departure)
            ORDER BY hour
        """
        
        hour_rows = query(conn, hour_query, hour_params if hour_params else None)
        chart_hour = {
            'labels': [f"{row['hour']}:00" for row in hour_rows],
            'values': [round(row['delay_rate'], 1) for row in hour_rows]
        } if hour_rows else {'labels': [], 'values': []}
        
        # ── CHART 3: Delay by Month (Filtered by airline) ─
        month_where = "1=1"
        month_params = []
        
        if airline_filter != 'ALL':
            month_where = "a.iata_code = %s"
            month_params.append(airline_filter)
        
        month_query = f"""
            SELECT 
                MONTH(wfr.flight_date) as month_num,
                COALESCE(AVG(wfr.is_delayed), 0) * 100 as delay_rate
            FROM weekly_flight_record wfr
            JOIN airline a ON wfr.airline_id = a.airline_id
            WHERE {month_where}
            GROUP BY MONTH(wfr.flight_date)
            ORDER BY month_num
        """
        
        month_rows = query(conn, month_query, month_params if month_params else None)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        chart_month = {
            'labels': [month_names[row['month_num']-1] for row in month_rows],
            'values': [round(row['delay_rate'], 1) for row in month_rows]
        } if month_rows else {'labels': [], 'values': []}
        
        # ── CHART 4: Delay by Day of Week (Filtered) ─────
        day_query = f"""
            SELECT 
                DAYOFWEEK(wfr.flight_date) as dow,
                COALESCE(AVG(wfr.is_delayed), 0) * 100 as delay_rate
            FROM weekly_flight_record wfr
            JOIN airline a ON wfr.airline_id = a.airline_id
            WHERE {month_where}
            GROUP BY DAYOFWEEK(wfr.flight_date)
            ORDER BY dow
        """
        
        day_rows = query(conn, day_query, month_params if month_params else None)
        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        
        chart_day = {
            'labels': [day_names[row['dow']-1] for row in day_rows],
            'values': [round(row['delay_rate'], 1) for row in day_rows]
        } if day_rows else {'labels': [], 'values': []}
        
        # ── TOP ROUTES ────────────────────────────────────
        top_routes = [
            {'origin': 'LAX', 'destination': 'JFK', 'delay_rate': 68},
            {'origin': 'ORD', 'destination': 'LGA', 'delay_rate': 62},
            {'origin': 'SFO', 'destination': 'LAX', 'delay_rate': 57},
            {'origin': 'ATL', 'destination': 'ORD', 'delay_rate': 54},
            {'origin': 'DFW', 'destination': 'LAX', 'delay_rate': 51},
            {'origin': 'JFK', 'destination': 'BOS', 'delay_rate': 49},
            {'origin': 'MIA', 'destination': 'JFK', 'delay_rate': 47},
            {'origin': 'DEN', 'destination': 'ORD', 'delay_rate': 44},
            {'origin': 'SEA', 'destination': 'SFO', 'delay_rate': 42},
            {'origin': 'LAS', 'destination': 'LAX', 'delay_rate': 38},
        ]
        
        close(conn)
        
        chart_data = {
            'airline': chart_airline if chart_airline['labels'] else {
                'labels': ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'VX'],
                'values': [42, 35, 48, 29, 38, 31, 55, 52, 22, 36]
            },
            'hour': chart_hour if chart_hour['labels'] else {
                'labels': [f"{h}:00" for h in range(6, 22)],
                'values': [18, 15, 12, 10, 11, 14, 19, 25, 28, 30, 32, 34, 35, 36, 38, 40]
            },
            'month': chart_month if chart_month['labels'] else {
                'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'values': [45, 38, 35, 32, 36, 40, 52, 48, 34, 30, 35, 50]
            },
            'day': chart_day if chart_day['labels'] else {
                'labels': ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
                'values': [35, 42, 34, 33, 38, 48, 30]
            },
        }
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to static data if database query fails
        stats = {
            'total_flights': 1284,
            'delay_rate': 38.4,
            'avg_delay': 34,
            'model_accuracy': 78,
        }
        chart_data = {
            'airline': {
                'labels': ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'VX'],
                'values': [42, 35, 48, 29, 38, 31, 55, 52, 22, 36],
            },
            'hour': {
                'labels': [f"{h}:00" for h in range(0, 24)],
                'values': [18, 15, 12, 10, 11, 14, 19, 25, 28, 30,
                          32, 34, 35, 36, 38, 40, 42, 45, 48, 50,
                          47, 44, 38, 28],
            },
            'month': {
                'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'values': [45, 38, 35, 32, 36, 40, 52, 48, 34, 30, 35, 50],
            },
            'day': {
                'labels': ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
                'values': [35, 42, 34, 33, 38, 48, 30],
            },
        }
        top_routes = [
            {'origin': 'LAX', 'destination': 'JFK', 'delay_rate': 68},
            {'origin': 'ORD', 'destination': 'LGA', 'delay_rate': 62},
            {'origin': 'SFO', 'destination': 'LAX', 'delay_rate': 57},
            {'origin': 'ATL', 'destination': 'ORD', 'delay_rate': 54},
            {'origin': 'DFW', 'destination': 'LAX', 'delay_rate': 51},
        ]
    
    # IMPORTANT: Make sure to include current_airline and current_month here
    return render_template('dashboard.html',
                           stats=stats,
                           chart_data=chart_data,
                           top_routes=top_routes,
                           airlines=get_airlines(),
                           current_airline=airline_filter,
                           current_month=month_filter)


# ── COMPARE ─────────────────────────────────────────────────
@app.route('/compare')
def compare():
    origin      = request.args.get('origin', '')
    destination = request.args.get('destination', '')
    flight_date = request.args.get('date', '')
    flights     = []

    if origin and destination and flight_date:
        try:
            dt  = datetime.strptime(flight_date, '%Y-%m-%d')
            dow = dt.isoweekday()
            mon = dt.month
        except Exception:
            dow, mon = 3, 6

        sample_flights = [
            ('AA101', 'AA', 'American Airlines',  '06:00', 315),
            ('DL202', 'DL', 'Delta Air Lines',    '09:30', 320),
            ('UA303', 'UA', 'United Airlines',    '12:45', 318),
            ('WN404', 'WN', 'Southwest Airlines', '15:00', 310),
            ('B6505', 'B6', 'JetBlue Airways',    '18:30', 325),
        ]

        for fn, code, name, dep, dur in sample_flights:
            hhmm = int(dep.replace(':', ''))
            dist = _distance_between_airports(origin, destination) or 2475
            pred = predict_delay(
                airline=code, origin=origin, destination=destination,
                month=mon, day_of_week=dow,
                scheduled_departure_hhmm=hhmm,
                departure_delay=0, distance=dist,
                scheduled_time=dur
            )
            flights.append({
                'flight_number':       fn,
                'airline_name':        name,
                'airline_code':        code,
                'scheduled_departure': dep,
                'scheduled_time':      dur,
                'origin':              origin,
                'destination':         destination,
                'delay_probability':   pred['delay_probability'],
                'risk_level':          pred['risk_level'],
                'ontime_rate':         random.randint(55, 85),
                'is_best':             False,
            })

        if flights:
            min(flights, key=lambda x: x['delay_probability'])['is_best'] = True

    return render_template('compare.html',
                           flights=flights,
                           airports=get_airports())


@app.route('/api/dashboard-stats')
def api_dashboard_stats():
    """API endpoint for dynamic dashboard filtering"""
    airline = request.args.get('airline', 'ALL')
    month = request.args.get('month', '0')
    
    try:
        conn = get_db()
        
        # Build dynamic query
        where_clause = "WHERE 1=1"
        params = []
        
        if airline != 'ALL':
            where_clause += " AND a.iata_code = %s"
            params.append(airline)
        
        if month != '0':
            where_clause += " AND MONTH(wfr.flight_date) = %s"
            params.append(month)
        
        # Get filtered stats
        query_str = f"""
            SELECT 
                COUNT(*) as total_flights,
                AVG(wfr.is_delayed) * 100 as delay_rate
            FROM weekly_flight_record wfr
            JOIN airline a ON wfr.airline_id = a.airline_id
            {where_clause}
        """
        
        result = query_one(conn, query_str, params if params else None)
        close(conn)
        
        return jsonify({
            'success': True,
            'stats': {
                'total_flights': int(result['total_flights']) if result else 0,
                'delay_rate': round(result['delay_rate'], 1) if result else 0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── LOGIN ────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        # ── Server-side validation ─────────────────────────
        ok, err = validate_email(email)
        if not ok:
            flash(err, 'error')
            return redirect(url_for('login'))
        if not password:
            flash('Password is required.', 'error')
            return redirect(url_for('login'))

        try:
            conn = get_db()
            user = query_one(conn,
                "SELECT * FROM user WHERE email = ?", (email,))
            close(conn)

            if user and check_password(password, user['password_hash']):
                session['user_id']  = user['user_id']
                session['username'] = user['username']
                conn = get_db()
                execute(conn,
                    "UPDATE user SET last_login = ? WHERE user_id = ?",
                    (str(datetime.utcnow()), user['user_id']))
                commit(conn)
                close(conn)
                flash('Welcome back, ' + user['username'] + '!', 'success')
                return redirect(url_for('profile'))
            else:
                flash('Incorrect email or password.', 'error')
        except Exception as e:
            flash('Login error: ' + str(e), 'error')

    return render_template('login.html')


# ── REGISTER ─────────────────────────────────────────────────
@app.route('/register', methods=['POST'])
def register():
    username        = request.form.get('username', '').strip()
    email           = request.form.get('email', '').strip().lower()
    password        = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    # ── Server-side validation (mirrors client-side rules) ──
    ok, err = validate_username(username)
    if not ok:
        flash(err, 'error')
        return redirect(url_for('login') + '?tab=register')

    ok, err = validate_email(email)
    if not ok:
        flash(err, 'error')
        return redirect(url_for('login') + '?tab=register')

    ok, err = validate_password(password)
    if not ok:
        flash(err, 'error')
        return redirect(url_for('login') + '?tab=register')

    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('login') + '?tab=register')

    pw_hash = hash_password(password)

    try:
        conn = get_db()
        uid  = execute(conn,
            "INSERT INTO user (username, email, password_hash, created_at) VALUES (?,?,?,?)",
            (username, email, pw_hash, str(datetime.utcnow())))
        commit(conn)
        close(conn)
        session['user_id']  = uid
        session['username'] = username
        flash('Account created! Welcome to SkyIQ.', 'success')
        return redirect(url_for('profile'))
    except Exception as e:
        if 'Duplicate' in str(e) or 'UNIQUE' in str(e):
            flash('That email is already registered. Please sign in instead.', 'error')
        else:
            flash('Registration error: ' + str(e), 'error')
        return redirect(url_for('login') + '?tab=register')


# ── LOGOUT ───────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ── DELETE ACCOUNT ───────────────────────────────────────────
@app.route('/delete-account', methods=['POST'])
def delete_account():
    """
    Permanently delete the current user's account after verifying their password.
    Cascades: user_feedback → prediction_log → search_history → user
    """
    if not session.get('user_id'):
        flash('Please sign in first.', 'info')
        return redirect(url_for('login'))

    password = request.form.get('password', '')
    uid      = int(session['user_id'])

    if not password:
        flash('Password is required to delete your account.', 'error')
        return redirect(url_for('profile'))

    try:
        conn = get_db()
        user = query_one(conn, "SELECT * FROM user WHERE user_id = ?", (uid,))
        close(conn)

        if not user or not check_password(password, user['password_hash']):
            flash('Incorrect password. Account was NOT deleted.', 'error')
            return redirect(url_for('profile'))

        # Cascade delete in FK order
        conn = get_db()
        execute(conn, "DELETE FROM user_feedback   WHERE user_id = ?", (uid,))
        execute(conn, "DELETE FROM prediction_log  WHERE user_id = ?", (uid,))
        execute(conn, "DELETE FROM search_history  WHERE user_id = ?", (uid,))
        execute(conn, "DELETE FROM user            WHERE user_id = ?", (uid,))
        commit(conn)
        close(conn)

        session.clear()
        flash('Your account has been permanently deleted. Sorry to see you go.', 'info')
        return redirect(url_for('index'))

    except Exception as e:
        flash('Error deleting account: ' + str(e), 'error')
        return redirect(url_for('profile'))


# ── PROFILE ──────────────────────────────────────────────────
def _stat_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


@app.route('/profile')
def profile():
    if not session.get('user_id'):
        flash('Please sign in to view your profile.', 'info')
        return redirect(url_for('login'))

    uid = int(session['user_id'])
    conn = None
    user = None
    history = []
    stats_row = None

    try:
        conn = get_db()
        try:
            row = query_one(conn,
                "SELECT * FROM user WHERE user_id = ?",
                (uid,))
            if row:
                user = dict(row)
                user['created_at'] = format_date(user.get('created_at'))
        except Exception:
            logger.exception("profile: user query failed (user_id=%s)", uid)

        try:
            history = query(conn, """
                SELECT sh.flight_number,
                       sh.queried_flight_date,
                       pl.delay_probability,
                       pl.risk_level,
                       uf.user_confirmed_delayed,
                       '' AS origin,
                       '' AS destination,
                       '' AS airline_code
                FROM search_history sh
                LEFT JOIN prediction_log pl
                       ON pl.prediction_id = (
                            SELECT pl2.prediction_id
                            FROM prediction_log pl2
                            WHERE pl2.user_id = sh.user_id
                              AND pl2.flight_number = sh.flight_number
                              AND pl2.predicted_for_date = sh.queried_flight_date
                              AND pl2.predicted_at <= sh.searched_at
                            ORDER BY pl2.predicted_at DESC, pl2.prediction_id DESC
                            LIMIT 1
                       )
                LEFT JOIN user_feedback uf
                       ON uf.prediction_id = pl.prediction_id
                WHERE sh.user_id = ?
                ORDER BY sh.searched_at DESC
                LIMIT 50
            """, (uid,))
        except Exception:
            logger.exception("profile: history query failed (user_id=%s)", uid)

        try:
            total_row = query_one(conn,
                "SELECT COUNT(*) AS n FROM search_history WHERE user_id = ?",
                (uid,))
            delayed_row = query_one(conn, """
                SELECT COALESCE(SUM(predicted_delayed), 0) AS n
                FROM prediction_log WHERE user_id = ?
            """, (uid,))
            ontime_row = query_one(conn, """
                SELECT COALESCE(SUM(CASE WHEN predicted_delayed = 0 THEN 1 ELSE 0 END), 0) AS n
                FROM prediction_log WHERE user_id = ?
            """, (uid,))
            stats_row = {
                'total':   total_row.get('n') if total_row else 0,
                'delayed': delayed_row.get('n') if delayed_row else 0,
                'ontime':  ontime_row.get('n') if ontime_row else 0,
            }
        except Exception:
            logger.exception("profile: stats query failed (user_id=%s)", uid)
    finally:
        if conn is not None:
            close(conn)

    if not user:
        user = {
            'username':   session.get('username', 'User'),
            'email':      '',
            'created_at': 'N/A'
        }

    stats = {
        'total_searches':      _stat_int(stats_row.get('total') if stats_row else None),
        'delayed_predictions': _stat_int(stats_row.get('delayed') if stats_row else None),
        'ontime_predictions':  _stat_int(stats_row.get('ontime') if stats_row else None),
    }

    return render_template('profile.html',
                           user=user,
                           history=history,
                           stats=stats)


# ── FEEDBACK ─────────────────────────────────────────────────
@app.route('/feedback', methods=['POST'])
def feedback():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    prediction_id  = request.form.get('prediction_id')
    actual_delayed = int(request.form.get('actual_delayed', 0))

    try:
        conn = get_db()
        execute(conn, """
            INSERT INTO user_feedback
              (prediction_id, user_id, user_confirmed_delayed, submitted_at)
            VALUES (?,?,?,?)
        """, (prediction_id, session['user_id'],
              actual_delayed, str(datetime.utcnow())))
        execute(conn, """
            UPDATE prediction_log
            SET actual_delayed = ?
            WHERE prediction_id = ?
        """, (actual_delayed, prediction_id))
        commit(conn)
        close(conn)
        flash('Thanks for your feedback!', 'success')
    except Exception:
        pass

    return redirect(url_for('profile'))


# ── RUN ──────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=DEBUG, port=PORT)
