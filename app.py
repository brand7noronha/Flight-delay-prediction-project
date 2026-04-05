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
import hashlib, logging, os, random, math
from datetime import datetime, date

logger = logging.getLogger(__name__)

# ── CONFIG & DB ────────────────────────────────────────────
from config import SECRET_KEY, DEBUG, PORT
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
# Sorted alphabetically so sklearn LabelEncoder produces the
# same integer mapping that was used during training.
_AIRLINE_CODES = sorted([
    '9E','AA','AS','B6','DL','EV','F9','HA',
    'MQ','NK','OO','UA','VX','WN','YX'
])
_AIRPORT_CODES = sorted([
    'ATL','BOS','CLT','DEN','DFW','DTW','EWR',
    'IAH','JFK','LAS','LAX','LGA','MCO','MIA',
    'MSP','ORD','PHX','SEA','SFO','SEA'
])
# Deduplicate while preserving sort
_AIRPORT_CODES = sorted(set(_AIRPORT_CODES))
_TIME_PERIODS  = sorted(['afternoon','evening','late_night','morning'])


def _build_label_encoders():
    """
    Build sklearn LabelEncoders for the four categorical features
    the regressor expects as integer codes.
    """
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

    # Look for the model in ./model/ first, then in the same directory
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
        print("⚠️  scikit-learn not installed — label encoders unavailable; "
              "falling back to Mock mode")

except FileNotFoundError as e:
    print(f"⚠️  {e}")
    print("   Place flight_delay_regressor.pkl in model/ (or same folder) "
          "to enable real predictions")
except ImportError as e:
    print(f"⚠️  Missing package: {e} — running in Mock mode")
    print("   Run: pip install joblib scikit-learn lightgbm pandas numpy")
except Exception as e:
    print(f"⚠️  Model load error: {e} — running in Mock mode")


# ── helper: safe label-encode a single value ───────────────
def _encode(col: str, value: str) -> int:
    """
    Return integer code for a categorical value.
    Falls back gracefully to 0 for unseen labels so a missing
    airline/airport code never crashes the prediction.
    """
    le = label_encoders.get(col)
    if le is None:
        return 0
    try:
        return int(le.transform([str(value)])[0])
    except ValueError:
        # Unseen label — use the first known class as a safe default
        return 0


# ── helper: delay-minutes → delay-probability ──────────────
def _minutes_to_prob(predicted_minutes: float) -> float:
    """
    Convert the regressor's predicted arrival-delay (in minutes)
    to a [0, 1] delay-probability using a logistic function
    centred at 15 min (the industry-standard "delayed" threshold).

      prob ≈ 0.09  when predicted_minutes = -30   (clearly early)
      prob ≈ 0.27  when predicted_minutes =   0   (right on time)
      prob = 0.50  when predicted_minutes =  15   (at the threshold)
      prob ≈ 0.69  when predicted_minutes =  30   (half-hour late)
      prob ≈ 0.88  when predicted_minutes =  60   (severely late)
    """
    return round(1.0 / (1.0 + math.exp(-(predicted_minutes - 15.0) / 20.0)), 4)


# ══════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# Uses real model if loaded, falls back to mock automatically
# ══════════════════════════════════════════════════════════
def predict_delay(airline, origin, destination, month,
                  day_of_week, scheduled_departure_hhmm,
                  departure_delay, distance, scheduled_time):
    """
    Main prediction function.
    Returns real ML prediction if the regressor is loaded,
    otherwise returns a rule-based mock prediction seamlessly.

    Return dict keys
    ────────────────
    is_delayed        bool   — True if likely delayed
    delay_probability float  — [0, 1] probability of delay
    predicted_delay_min float — raw regressor output (minutes), or None
    risk_level        str   — 'Low' / 'Medium' / 'High'
    departure_hour    int
    time_period       str
    model_name        str
    message           str
    """
    dep_hour = int(str(scheduled_departure_hhmm).zfill(4)[:2]) \
               if scheduled_departure_hhmm else 12

    if   dep_hour < 6:  time_period = 'late_night'
    elif dep_hour < 12: time_period = 'morning'
    elif dep_hour < 18: time_period = 'afternoon'
    else:               time_period = 'evening'

    # ── REAL MODEL PATH ────────────────────────────────────
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
            pred = prob >= 0.50                         # delayed if ≥50%
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

    # ── MOCK FALLBACK PATH ─────────────────────────────────
    base_prob = 0.28
    if dep_hour >= 18:                 base_prob += 0.20
    if dep_hour >= 21:                 base_prob += 0.10
    if float(departure_delay) > 0:     base_prob += 0.25
    if month in [7, 12, 1]:           base_prob += 0.10
    if day_of_week in [5, 1]:         base_prob += 0.05

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


# ── DATE HELPER ─────────────────────────────────────────────
def format_date(value):
    """
    Safely format a date/datetime whether it comes as a Python
    datetime object or a MySQL string.  Returns e.g. 'Jan 2024'
    """
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
        flight_no   = form.get('flight_no', '').strip().upper() or 'N/A'
        airline     = form.get('airline', 'AA')
        origin      = form.get('origin', 'LAX')
        destination = form.get('destination', 'JFK')
        flight_date = form.get('flight_date') or str(date.today())
        sched_dep   = form.get('scheduled_departure', '12:00').replace(':', '')
        distance    = form.get('distance', 500)
        dep_delay   = form.get('departure_delay', 0)
        sched_time  = form.get('scheduled_time', 120)

        try:
            dt    = datetime.strptime(flight_date, '%Y-%m-%d')
            month = dt.month
            dow   = dt.isoweekday()
        except Exception:
            month, dow = 6, 3

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

        prediction.update({
            'flight_number':       flight_no,
            'airline':             airline,
            'origin':              origin,
            'destination':         destination,
            'flight_date':         flight_date,
            'scheduled_departure': form.get('scheduled_departure', '12:00'),
            'distance':            distance,
            'departure_delay':     dep_delay,
        })

        # ── Historical stats from DB; fall back to mock ────
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

        # ── Persist to DB if user is logged in ────────────
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


# ── DASHBOARD ───────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    stats = {
        'total_flights':  1284,
        'delay_rate':     38.4,
        'avg_delay':      34,
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
    return render_template('dashboard.html',
                           stats=stats,
                           chart_data=chart_data,
                           top_routes=MOCK_TOP_ROUTES,
                           airlines=get_airlines())


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
            pred = predict_delay(
                airline=code, origin=origin, destination=destination,
                month=mon, day_of_week=dow,
                scheduled_departure_hhmm=hhmm,
                departure_delay=0, distance=2475,
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


# ── LOGIN ────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
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
    username = request.form.get('username', '').strip()
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    pw_hash  = hash_password(password)

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
            flash('Email already registered. Please sign in.', 'error')
        else:
            flash('Registration error: ' + str(e), 'error')
        return redirect(url_for('login'))


# ── LOGOUT ───────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


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
            # One prediction per search row: same flight+date can appear many times in
            # prediction_log; a plain join multiplies rows (e.g. 3 searches -> 9 rows).
            # Pick the latest prediction at or before this search's timestamp.
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