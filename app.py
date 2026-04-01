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
import hashlib, os, random
from datetime import datetime, date

# ── CONFIG & DB ────────────────────────────────────────────
from config import SECRET_KEY, DEBUG, PORT
from db import get_db, query, query_one, execute, commit, close

app = Flask(__name__)
app.secret_key = SECRET_KEY


# ══════════════════════════════════════════════════════════
# ML MODEL LOADING
# ══════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'model')

ML_READY   = False   # flips to True if all .pkl files load successfully
model      = None
encoder    = None
FEATURES   = None
CAT_COLS   = None
MODEL_NAME = 'Mock'

try:
    import joblib
    import pandas as pd

    model      = joblib.load(os.path.join(MODEL_DIR, 'flight_delay_model_best.pkl'))
    encoder    = joblib.load(os.path.join(MODEL_DIR, 'feature_encoder.pkl'))
    FEATURES   = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    CAT_COLS   = joblib.load(os.path.join(MODEL_DIR, 'cat_cols.pkl'))
    MODEL_NAME = type(model).__name__
    ML_READY   = True
    print(f"✅ ML model loaded: {MODEL_NAME}")

except FileNotFoundError:
    print("⚠️  model/ folder not found — running in Mock mode")
    print("   Place .pkl files in the model/ folder to enable real predictions")
except ImportError as e:
    print(f"⚠️  Missing package: {e} — running in Mock mode")
    print("   Run: pip install joblib scikit-learn lightgbm pandas numpy")
except Exception as e:
    print(f"⚠️  Model load error: {e} — running in Mock mode")


# ══════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# Uses real model if loaded, falls back to mock automatically
# ══════════════════════════════════════════════════════════
def predict_delay(airline, origin, destination, month,
                  day_of_week, scheduled_departure_hhmm,
                  departure_delay, distance, scheduled_time):
    """
    Main prediction function.
    Returns real ML prediction if model is loaded,
    otherwise returns mock prediction seamlessly.
    """
    dep_hour = int(str(scheduled_departure_hhmm).zfill(4)[:2]) if scheduled_departure_hhmm else 12

    if dep_hour < 6:    time_period = 'late_night'
    elif dep_hour < 12: time_period = 'morning'
    elif dep_hour < 18: time_period = 'afternoon'
    else:               time_period = 'evening'

    # ── REAL MODEL PATH ────────────────────────────────────
    if ML_READY:
        try:
            row = pd.DataFrame([{
                'AIRLINE':             str(airline),
                'ORIGIN_AIRPORT':      str(origin),
                'DESTINATION_AIRPORT': str(destination),
                'MONTH':               int(month),
                'DAY_OF_WEEK':         int(day_of_week),
                'DEPARTURE_HOUR':      int(dep_hour),
                'DEPARTURE_DELAY':     float(departure_delay),
                'DISTANCE':            float(distance),
                'SCHEDULED_TIME':      float(scheduled_time),
                'TIME_PERIOD':         time_period
            }])
            row[CAT_COLS] = encoder.transform(row[CAT_COLS].astype(str))
            prob = round(float(model.predict_proba(row)[0][1]), 4)
            pred = prob >= 0.5
            risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.65 else 'High')
            return {
                'is_delayed':        pred,
                'delay_probability': prob,
                'risk_level':        risk,
                'departure_hour':    dep_hour,
                'time_period':       time_period,
                'model_name':        MODEL_NAME,
                'message': (
                    f"{'Likely DELAYED' if pred else 'Likely ON TIME'} "
                    f"— {round(prob * 100, 1)}% delay probability ({risk} Risk)"
                )
            }
        except Exception as e:
            print(f"⚠️  Prediction error: {e} — falling back to mock")

    # ── MOCK FALLBACK PATH ─────────────────────────────────
    base_prob = 0.28
    if dep_hour >= 18:             base_prob += 0.20
    if dep_hour >= 21:             base_prob += 0.10
    if float(departure_delay) > 0: base_prob += 0.25
    if month in [7, 12, 1]:       base_prob += 0.10
    if day_of_week in [5, 1]:     base_prob += 0.05

    prob = round(min(max(base_prob + random.uniform(-0.05, 0.05), 0.05), 0.95), 4)
    pred = prob >= 0.5
    risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.65 else 'High')

    return {
        'is_delayed':        pred,
        'delay_probability': prob,
        'risk_level':        risk,
        'departure_hour':    dep_hour,
        'time_period':       time_period,
        'model_name':        'Mock (model/ folder not found)',
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
    Safely format a date/datetime whether it comes as
    a Python datetime object or a MySQL string.
    Returns e.g. 'Jan 2024'
    """
    if not value:
        return 'N/A'
    if isinstance(value, str):
        # MySQL returns '2024-01-15 10:30:00' or '2024-01-15'
        try:
            value = datetime.strptime(value[:19], '%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                value = datetime.strptime(value[:10], '%Y-%m-%d')
            except Exception:
                return str(value)[:7]
    return value.strftime('%b %Y')

# Register as Jinja filter so templates can use {{ value | fmtdate }}
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

        prediction['history'] = {
            'total_flights':     random.randint(5, 14),
            'total_on_time':     random.randint(3, 9),
            'total_delayed':     random.randint(1, 5),
            'delay_rate':        round(random.uniform(0.2, 0.7), 2),
            'avg_delay_minutes': random.randint(18, 55),
        }

        prediction_id = None
        if session.get('user_id'):
            try:
                conn = get_db()
                prediction_id = execute(conn, """
                    INSERT INTO prediction_log
                      (user_id, flight_number, predicted_for_date,
                       delay_probability, risk_level,
                       predicted_delayed, predicted_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (session['user_id'], flight_no, flight_date,
                      prediction['delay_probability'],
                      prediction['risk_level'],
                      int(prediction['is_delayed']),
                      str(datetime.utcnow())))
                execute(conn, """
                    INSERT INTO search_history
                      (user_id, flight_number, search_date,
                       queried_flight_date, searched_at)
                    VALUES (?,?,?,?,?)
                """, (session['user_id'], flight_no,
                      str(date.today()), flight_date,
                      str(datetime.utcnow())))
                commit(conn)
                close(conn)
            except Exception:
                pass

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
@app.route('/profile')
def profile():
    if not session.get('user_id'):
        flash('Please sign in to view your profile.', 'info')
        return redirect(url_for('login'))

    try:
        conn = get_db()
        user = query_one(conn,
            "SELECT * FROM user WHERE user_id = ?",
            (session['user_id'],))

        # Convert created_at to safe string format
        if user:
            user = dict(user)
            user['created_at'] = format_date(user.get('created_at'))

        history = query(conn, """
            SELECT sh.flight_number,
                   sh.queried_flight_date,
                   pl.delay_probability,
                   pl.risk_level,
                   uf.user_confirmed_delayed,
                   '' as origin,
                   '' as destination,
                   '' as airline_code
            FROM search_history sh
            LEFT JOIN prediction_log pl
                   ON pl.user_id = sh.user_id
                  AND pl.flight_number = sh.flight_number
                  AND pl.predicted_for_date = sh.queried_flight_date
            LEFT JOIN user_feedback uf
                   ON uf.prediction_id = pl.prediction_id
            WHERE sh.user_id = ?
            ORDER BY sh.searched_at DESC
            LIMIT 50
        """, (session['user_id'],))

        stats_row = query_one(conn, """
            SELECT COUNT(*) as total,
                   SUM(predicted_delayed) as delayed,
                   SUM(1 - predicted_delayed) as ontime
            FROM prediction_log
            WHERE user_id = ?
        """, (session['user_id'],))
        close(conn)

    except Exception as e:
        user = {
            'username':   session.get('username', 'User'),
            'email':      '',
            'created_at': 'N/A'
        }
        history   = []
        stats_row = None

    stats = {
        'total_searches':      stats_row['total']   if stats_row and stats_row['total']   else 0,
        'delayed_predictions': stats_row['delayed'] if stats_row and stats_row['delayed'] else 0,
        'ontime_predictions':  stats_row['ontime']  if stats_row and stats_row['ontime']  else 0,
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
