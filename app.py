"""
app.py — SkyIQ Flask Backend (Frontend-Only Mode)
Flight Delay Prediction System
MCA Group 3 — Apa Mestry (2512) & Brandon Noronha (2514)

FRONTEND MODE — No ML, no numpy, no pandas, no sklearn.
All predictions return realistic mock data so every page is fully testable.
Plug in the real model later by replacing the mock_predict() function.

Install (only 2 packages needed!):
    pip install flask requests
    python app.py

Open: http://localhost:5000
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
import sqlite3, hashlib, os, random
from datetime import datetime, date

app = Flask(__name__)
app.secret_key = 'skyiq-secret-change-in-production'

# ── PATHS ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'database', 'flights.db')

# ══════════════════════════════════════════════════════════
# MOCK PREDICTION
# Replace this entire function later with real model
# ══════════════════════════════════════════════════════════
def mock_predict(airline, origin, destination, month,
                 day_of_week, scheduled_departure_hhmm,
                 departure_delay, distance, scheduled_time):
    dep_hour = int(str(scheduled_departure_hhmm).zfill(4)[:2]) if scheduled_departure_hhmm else 12

    base_prob = 0.28
    if dep_hour >= 18:              base_prob += 0.20
    if dep_hour >= 21:              base_prob += 0.10
    if float(departure_delay) > 0:  base_prob += 0.25
    if month in [7, 12, 1]:        base_prob += 0.10
    if day_of_week in [5, 1]:      base_prob += 0.05

    prob = round(min(max(base_prob + random.uniform(-0.05, 0.05), 0.05), 0.95), 4)
    pred = prob >= 0.5
    risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.65 else 'High')

    if dep_hour < 6:    time_period = 'late_night'
    elif dep_hour < 12: time_period = 'morning'
    elif dep_hour < 18: time_period = 'afternoon'
    else:               time_period = 'evening'

    return {
        'is_delayed':        pred,
        'delay_probability': prob,
        'risk_level':        risk,
        'departure_hour':    dep_hour,
        'time_period':       time_period,
        'model_name':        'Mock (ML coming soon)',
        'message': (
            f"{'Likely DELAYED' if pred else 'Likely ON TIME'} "
            f"— {round(prob*100,1)}% delay probability ({risk} Risk)"
        )
    }


# ── PASSWORD HELPERS ────────────────────────────────────────
def hash_password(password: str) -> str:
    salt   = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def check_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(':', 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


# ── DB HELPER ───────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── MOCK LOOKUP DATA ────────────────────────────────────────
MOCK_AIRLINES = [
    ('AA', 'American Airlines'),   ('DL', 'Delta Air Lines'),
    ('UA', 'United Airlines'),     ('WN', 'Southwest Airlines'),
    ('B6', 'JetBlue Airways'),     ('AS', 'Alaska Airlines'),
    ('NK', 'Spirit Airlines'),     ('F9', 'Frontier Airlines'),
    ('HA', 'Hawaiian Airlines'),   ('VX', 'Virgin America'),
]

MOCK_AIRPORTS = [
    ('ATL', 'Hartsfield-Jackson Atlanta',     'Atlanta'),
    ('LAX', 'Los Angeles International',      'Los Angeles'),
    ('ORD', "O'Hare International",           'Chicago'),
    ('DFW', 'Dallas/Fort Worth International','Dallas'),
    ('DEN', 'Denver International',           'Denver'),
    ('JFK', 'John F. Kennedy International',  'New York'),
    ('SFO', 'San Francisco International',    'San Francisco'),
    ('SEA', 'Seattle-Tacoma International',   'Seattle'),
    ('LAS', 'Harry Reid International',       'Las Vegas'),
    ('MCO', 'Orlando International',          'Orlando'),
    ('MIA', 'Miami International',            'Miami'),
    ('BOS', 'Logan International',            'Boston'),
    ('MSP', 'Minneapolis-Saint Paul',         'Minneapolis'),
    ('PHX', 'Phoenix Sky Harbor',             'Phoenix'),
    ('EWR', 'Newark Liberty International',   'Newark'),
]

MOCK_TOP_ROUTES = [
    {'origin':'LAX','destination':'JFK','delay_rate':68},
    {'origin':'ORD','destination':'LGA','delay_rate':62},
    {'origin':'SFO','destination':'LAX','delay_rate':57},
    {'origin':'ATL','destination':'ORD','delay_rate':54},
    {'origin':'DFW','destination':'LAX','delay_rate':51},
    {'origin':'JFK','destination':'BOS','delay_rate':49},
    {'origin':'MIA','destination':'JFK','delay_rate':47},
    {'origin':'DEN','destination':'ORD','delay_rate':44},
    {'origin':'SEA','destination':'SFO','delay_rate':42},
    {'origin':'LAS','destination':'LAX','delay_rate':38},
]

def get_airlines():
    try:
        db   = get_db()
        rows = db.execute(
            "SELECT iata_code, airline_name FROM airline ORDER BY airline_name"
        ).fetchall()
        db.close()
        if rows: return [(r['iata_code'], r['airline_name']) for r in rows]
    except Exception:
        pass
    return MOCK_AIRLINES

def get_airports():
    try:
        db   = get_db()
        rows = db.execute(
            "SELECT iata_code, airport_name, city FROM airport ORDER BY city"
        ).fetchall()
        db.close()
        if rows: return [(r['iata_code'], r['airport_name'], r['city']) for r in rows]
    except Exception:
        pass
    return MOCK_AIRPORTS


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


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

        prediction = mock_predict(
            airline=airline, origin=origin, destination=destination,
            month=month, day_of_week=dow,
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
                db  = get_db()
                cur = db.execute("""
                    INSERT INTO prediction_log
                      (user_id, flight_number, predicted_for_date,
                       delay_probability, risk_level,
                       predicted_delayed, predicted_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (session['user_id'], flight_no, flight_date,
                      prediction['delay_probability'],
                      prediction['risk_level'],
                      int(prediction['is_delayed']),
                      datetime.utcnow()))
                prediction_id = cur.lastrowid
                db.execute("""
                    INSERT INTO search_history
                      (user_id, flight_number, search_date,
                       queried_flight_date, searched_at)
                    VALUES (?,?,?,?,?)
                """, (session['user_id'], flight_no,
                      date.today(), flight_date, datetime.utcnow()))
                db.commit(); db.close()
            except Exception:
                pass

        prediction['prediction_id'] = prediction_id
        return render_template('result.html', result=prediction)

    return render_template('predict.html',
                           airlines=airlines,
                           airports=airports,
                           prefill=prefill)


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
            'labels': ['AA','DL','UA','WN','B6','AS','NK','F9','HA','VX'],
            'values': [42, 35, 48, 29, 38, 31, 55, 52, 22, 36],
        },
        'hour': {
            'labels': [f"{h}:00" for h in range(0, 24)],
            'values': [18,15,12,10,11,14,19,25,28,30,
                       32,34,35,36,38,40,42,45,48,50,
                       47,44,38,28],
        },
        'month': {
            'labels': ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec'],
            'values': [45,38,35,32,36,40,52,48,34,30,35,50],
        },
        'day': {
            'labels': ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],
            'values': [35,42,34,33,38,48,30],
        },
    }
    return render_template('dashboard.html',
                           stats=stats,
                           chart_data=chart_data,
                           top_routes=MOCK_TOP_ROUTES,
                           airlines=get_airlines())


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
            ('AA101','AA','American Airlines',  '06:00',315),
            ('DL202','DL','Delta Air Lines',    '09:30',320),
            ('UA303','UA','United Airlines',    '12:45',318),
            ('WN404','WN','Southwest Airlines', '15:00',310),
            ('B6505','B6','JetBlue Airways',    '18:30',325),
        ]
        for fn, code, name, dep, dur in sample_flights:
            hhmm = int(dep.replace(':',''))
            pred = mock_predict(
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


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        try:
            db   = get_db()
            user = db.execute(
                "SELECT * FROM user WHERE email=?", (email,)
            ).fetchone()
            db.close()
            if user and check_password(password, user['password_hash']):
                session['user_id']  = user['user_id']
                session['username'] = user['username']
                db = get_db()
                db.execute("UPDATE user SET last_login=? WHERE user_id=?",
                           (datetime.utcnow(), user['user_id']))
                db.commit(); db.close()
                flash('Welcome back, ' + user['username'] + '!', 'success')
                return redirect(url_for('profile'))
            else:
                flash('Incorrect email or password.', 'error')
        except Exception:
            flash('Database not set up yet. Run: python init_db.py', 'error')
    return render_template('login.html')


@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username','').strip()
    email    = request.form.get('email','').strip().lower()
    password = request.form.get('password','')
    pw_hash  = hash_password(password)
    try:
        db  = get_db()
        cur = db.execute(
            "INSERT INTO user (username,email,password_hash,created_at) VALUES (?,?,?,?)",
            (username, email, pw_hash, datetime.utcnow())
        )
        db.commit()
        session['user_id']  = cur.lastrowid
        session['username'] = username
        db.close()
        flash('Account created! Welcome to SkyIQ.', 'success')
        return redirect(url_for('profile'))
    except sqlite3.IntegrityError:
        flash('Email already registered. Please sign in.', 'error')
        return redirect(url_for('login'))
    except Exception:
        flash('Database not set up yet. Run: python init_db.py', 'error')
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/profile')
def profile():
    if not session.get('user_id'):
        flash('Please sign in to view your profile.', 'info')
        return redirect(url_for('login'))
    try:
        db      = get_db()
        user    = db.execute(
            "SELECT * FROM user WHERE user_id=?",
            (session['user_id'],)
        ).fetchone()
        history = db.execute("""
            SELECT sh.flight_number, sh.queried_flight_date,
                   pl.delay_probability, pl.risk_level,
                   uf.user_confirmed_delayed,
                   '' as origin, '' as destination, '' as airline_code
            FROM search_history sh
            LEFT JOIN prediction_log pl
                   ON pl.user_id=sh.user_id
                  AND pl.flight_number=sh.flight_number
                  AND pl.predicted_for_date=sh.queried_flight_date
            LEFT JOIN user_feedback uf ON uf.prediction_id=pl.prediction_id
            WHERE sh.user_id=?
            ORDER BY sh.searched_at DESC LIMIT 50
        """, (session['user_id'],)).fetchall()
        stats_row = db.execute("""
            SELECT COUNT(*) as total,
                   SUM(predicted_delayed) as delayed,
                   SUM(1-predicted_delayed) as ontime
            FROM prediction_log WHERE user_id=?
        """, (session['user_id'],)).fetchone()
        db.close()
    except Exception:
        user      = {'username': session.get('username','User'),
                     'email':'', 'created_at': datetime.utcnow()}
        history   = []
        stats_row = None

    stats = {
        'total_searches':      stats_row['total']   if stats_row else 0,
        'delayed_predictions': stats_row['delayed'] if stats_row else 0,
        'ontime_predictions':  stats_row['ontime']  if stats_row else 0,
    }
    return render_template('profile.html',
                           user=dict(user),
                           history=[dict(h) for h in history],
                           stats=stats)


@app.route('/feedback', methods=['POST'])
def feedback():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    prediction_id  = request.form.get('prediction_id')
    actual_delayed = int(request.form.get('actual_delayed', 0))
    try:
        db = get_db()
        db.execute("""
            INSERT OR REPLACE INTO user_feedback
              (prediction_id, user_id, user_confirmed_delayed, submitted_at)
            VALUES (?,?,?,?)
        """, (prediction_id, session['user_id'],
              actual_delayed, datetime.utcnow()))
        db.execute("""
            UPDATE prediction_log SET actual_delayed=?
            WHERE prediction_id=?
        """, (actual_delayed, prediction_id))
        db.commit(); db.close()
        flash('Thanks for your feedback!', 'success')
    except Exception:
        pass
    return redirect(url_for('profile'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
