"""
seed_flight_data.py — Populate weekly_flight_record with realistic flight data
Run AFTER init_db_mysql.py:
    python seed_flight_data.py
"""

import random
import pymysql
from datetime import datetime, timedelta
from config import MYSQL_CONFIG, DB_TYPE

# Airport codes that exist in your database (from init_db_mysql.py)
AIRPORT_CODES = [
    'ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 
    'LAS', 'MCO', 'MIA', 'BOS', 'MSP', 'PHX', 'EWR', 'IAH', 
    'CLT', 'LGA', 'DTW', 'PHL'
]

# Airline codes that exist in your database (from init_db_mysql.py)
AIRLINE_CODES = [
    'AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'VX',
    'OO', 'MQ', 'EV', 'YX', '9E'
]

# Realistic distances between major airports (miles)
def get_distance(origin, destination):
    """Get realistic distance between airports"""
    # Common route distances (simplified but realistic)
    routes = {
        ('LAX', 'JFK'): 2475, ('JFK', 'LAX'): 2475,
        ('ORD', 'LGA'): 730, ('LGA', 'ORD'): 730,
        ('SFO', 'LAX'): 337, ('LAX', 'SFO'): 337,
        ('ATL', 'ORD'): 606, ('ORD', 'ATL'): 606,
        ('DFW', 'LAX'): 1235, ('LAX', 'DFW'): 1235,
        ('JFK', 'BOS'): 186, ('BOS', 'JFK'): 186,
        ('MIA', 'JFK'): 1092, ('JFK', 'MIA'): 1092,
        ('DEN', 'ORD'): 888, ('ORD', 'DEN'): 888,
        ('SEA', 'SFO'): 679, ('SFO', 'SEA'): 679,
        ('LAS', 'LAX'): 236, ('LAX', 'LAS'): 236,
    }
    
    key = (origin, destination)
    if key in routes:
        return routes[key]
    
    # Default distance based on region
    west_coast = ['LAX', 'SFO', 'SEA', 'LAS', 'PHX', 'DEN']
    east_coast = ['JFK', 'LGA', 'BOS', 'EWR', 'MIA', 'ATL', 'CLT', 'PHL', 'MCO']
    
    if origin in west_coast and destination in east_coast:
        return random.randint(2000, 2800)
    elif origin in east_coast and destination in west_coast:
        return random.randint(2000, 2800)
    elif origin in west_coast and destination in west_coast:
        return random.randint(300, 1000)
    elif origin in east_coast and destination in east_coast:
        return random.randint(300, 1200)
    else:
        return random.randint(500, 1500)

def generate_flight_records(num_records=5000):
    """Generate realistic flight records"""
    records = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    for i in range(num_records):
        # Random date within 2024
        days_offset = random.randint(0, date_range)
        flight_date = start_date + timedelta(days=days_offset)
        month = flight_date.month
        day_of_week = flight_date.isoweekday()
        
        # Random airline
        airline_code = random.choice(AIRLINE_CODES)
        
        # Random route (different airports)
        origin = random.choice(AIRPORT_CODES)
        destination = random.choice([a for a in AIRPORT_CODES if a != origin])
        
        # Get distance
        distance = get_distance(origin, destination)
        
        # Scheduled departure hour (realistic distribution)
        # Peak hours: 6-9am, 4-7pm
        hour_weights = [2, 1, 1, 1, 2, 8, 12, 10, 8, 6, 5, 5, 5, 5, 6, 8, 10, 12, 10, 8, 5, 3, 2, 2]
        scheduled_hour = random.choices(list(range(24)), weights=hour_weights, k=1)[0]
        scheduled_minute = random.choice([0, 15, 30, 45])
        scheduled_departure = f"{scheduled_hour:02d}:{scheduled_minute:02d}"
        
        # Scheduled flight time (minutes) based on distance
        # Average speed ~450 mph, plus taxi time
        scheduled_time = int((distance / 450) * 60) + random.randint(15, 45)
        scheduled_time = max(45, min(600, scheduled_time))
        
        # Calculate delay probability based on multiple factors
        delay_prob = 0.20  # base 20% delay rate
        
        # Time of day factor
        if scheduled_hour >= 16 and scheduled_hour <= 19:  # evening peak
            delay_prob += 0.25
        elif scheduled_hour >= 20 or scheduled_hour <= 5:  # night/early morning
            delay_prob += 0.10
        elif scheduled_hour >= 7 and scheduled_hour <= 9:  # morning peak
            delay_prob += 0.15
            
        # Day of week factor
        if day_of_week == 5:  # Friday
            delay_prob += 0.12
        elif day_of_week == 1:  # Monday
            delay_prob += 0.10
        elif day_of_week in [6, 7]:  # Weekend
            delay_prob -= 0.05
            
        # Season factor
        if month in [6, 7, 8]:  # Summer thunderstorms
            delay_prob += 0.12
        elif month in [12, 1, 2]:  # Winter weather
            delay_prob += 0.15
        elif month in [3, 4, 5]:  # Spring
            delay_prob += 0.05
            
        # Distance factor
        if distance > 2000:  # Long haul
            delay_prob += 0.10
        elif distance < 500:  # Short hop
            delay_prob -= 0.05
            
        # Airline reliability factor
        reliable_airlines = ['DL', 'WN', 'AS']
        unreliable_airlines = ['NK', 'F9', 'HA', 'VX']
        
        if airline_code in reliable_airlines:
            delay_prob -= 0.08
        elif airline_code in unreliable_airlines:
            delay_prob += 0.12
            
        # Clamp probability
        delay_prob = max(0.05, min(0.85, delay_prob))
        
        # Determine if flight is delayed
        is_delayed = random.random() < delay_prob
        
        # Calculate actual delay minutes
        if is_delayed:
            # Longer delays less common
            delay_type = random.random()
            if delay_type < 0.6:  # 60%: minor delay (15-45 min)
                delay_minutes = random.randint(15, 45)
            elif delay_type < 0.85:  # 25%: moderate delay (46-90 min)
                delay_minutes = random.randint(46, 90)
            else:  # 15%: major delay (91-240 min)
                delay_minutes = random.randint(91, 240)
        else:
            # On-time or early
            delay_minutes = random.randint(-15, 14)
            
        # Departure delay (often similar to arrival delay)
        if is_delayed and delay_minutes > 0:
            departure_delay = delay_minutes - random.randint(-10, 20)
            departure_delay = max(0, min(departure_delay, delay_minutes + 15))
        else:
            departure_delay = random.randint(-10, 5)
            
        records.append({
            'flight_number': f"{airline_code}{random.randint(100, 9999)}",
            'airline_code': airline_code,
            'origin': origin,
            'destination': destination,
            'flight_date': flight_date.strftime('%Y-%m-%d'),
            'scheduled_departure': scheduled_departure,
            'distance': distance,
            'scheduled_time': scheduled_time,
            'departure_delay': max(-10, departure_delay),
            'arrival_delay': delay_minutes,
            'is_delayed': 1 if delay_minutes > 15 else 0,
            'month': month,
            'day_of_week': day_of_week,
            'departure_hour': scheduled_hour
        })
    
    return records

def seed_flight_data():
    """Insert flight records into weekly_flight_record table"""
    if DB_TYPE != 'mysql':
        print("⚠️  This script requires MySQL. Set DB_TYPE='mysql' in config.py")
        return
    
    print("📊 Connecting to MySQL...")
    conn = pymysql.connect(
        host=MYSQL_CONFIG['host'],
        user=MYSQL_CONFIG['user'],
        password=MYSQL_CONFIG['password'],
        database=MYSQL_CONFIG['database'],
        charset='utf8mb4'
    )
    cursor = conn.cursor()
    
    # Temporarily disable ONLY_FULL_GROUP_BY
    print("⚙️  Configuring MySQL session...")
    cursor.execute("SET SESSION sql_mode = ''")
    
    # Check if we already have data
    cursor.execute("SELECT COUNT(*) as cnt FROM weekly_flight_record")
    result = cursor.fetchone()
    existing_count = result[0] if result else 0
    
    if existing_count > 0:
        print(f"⚠️  Found {existing_count} existing records in weekly_flight_record")
        response = input("Do you want to clear existing data and reseed? (y/n): ")
        if response.lower() != 'y':
            print("❌ Seeding cancelled")
            cursor.close()
            conn.close()
            return
        print("🗑️  Clearing existing records...")
        cursor.execute("TRUNCATE TABLE weekly_flight_record")
        cursor.execute("TRUNCATE TABLE flight_aggregate")
        conn.commit()
    
    # Generate flight records
    print("✈️  Generating flight records...")
    num_records = 5000  # Adjust this number as needed
    records = generate_flight_records(num_records)
    print(f"✅ Generated {len(records)} records")
    
    # Get airline ID mapping
    cursor.execute("SELECT iata_code, airline_id FROM airline")
    airline_map = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Insert records
    print("💾 Inserting into database...")
    insert_sql = """
        INSERT INTO weekly_flight_record 
        (flight_number, airline_id, flight_date, scheduled_departure, 
         departure_delay_min, arrival_delay_min, is_delayed, scheduled_time, is_cancelled)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0)
    """
    
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        for rec in batch:
            airline_id = airline_map.get(rec['airline_code'])
            if airline_id:
                cursor.execute(insert_sql, (
                    rec['flight_number'],
                    airline_id,
                    rec['flight_date'],
                    rec['scheduled_departure'],
                    rec['departure_delay'],
                    rec['arrival_delay'],
                    rec['is_delayed'],
                    rec['scheduled_time']
                ))
        
        conn.commit()
        print(f"  📦 Inserted {min(i+batch_size, len(records))} of {len(records)} records")
    
    # Update flight_aggregate table - Fixed version
    print("📈 Updating flight aggregates...")
    
    # First, clear existing aggregates
    cursor.execute("TRUNCATE TABLE flight_aggregate")
    
    # Insert aggregates using a simpler, more compatible query
    aggregate_query = """
        INSERT INTO flight_aggregate 
        (flight_number, airline_id, week_start_date, week_end_date, 
         total_flights, total_delayed, total_on_time, delay_rate, avg_delay_minutes)
        SELECT 
            flight_number,
            airline_id,
            MIN(DATE_SUB(flight_date, INTERVAL WEEKDAY(flight_date) DAY)) as week_start,
            MAX(DATE_ADD(DATE_SUB(flight_date, INTERVAL WEEKDAY(flight_date) DAY), INTERVAL 6 DAY)) as week_end,
            COUNT(*) as total_flights,
            SUM(is_delayed) as total_delayed,
            SUM(CASE WHEN is_delayed = 0 THEN 1 ELSE 0 END) as total_on_time,
            (SUM(is_delayed) / COUNT(*)) * 100 as delay_rate,
            AVG(CASE WHEN arrival_delay_min > 0 THEN arrival_delay_min ELSE 0 END) as avg_delay_minutes
        FROM weekly_flight_record
        GROUP BY flight_number, airline_id, DATE_SUB(flight_date, INTERVAL WEEKDAY(flight_date) DAY)
    """
    
    cursor.execute(aggregate_query)
    conn.commit()
    
    # Show summary
    cursor.execute("SELECT COUNT(*) as total FROM weekly_flight_record")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT flight_number) as distinct_flights FROM weekly_flight_record")
    distinct_flights = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(is_delayed) * 100 as overall_delay FROM weekly_flight_record")
    overall_delay = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) as agg_count FROM flight_aggregate")
    agg_count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*50)
    print("✅ DATABASE SEEDING COMPLETE!")
    print("="*50)
    print(f"📊 Total flight records: {total_records:,}")
    print(f"✈️  Unique flights: {distinct_flights}")
    print(f"⏰ Overall delay rate: {overall_delay:.1f}%")
    print(f"📊 Aggregate records: {agg_count}")
    print(f"📅 Date range: Jan 2024 - Dec 2024")
    print("\n🎯 Next steps:")
    print("   1. Restart your Flask app: python app.py")
    print("   2. Visit the Dashboard to see dynamic data")
    print("   3. Try filtering by airline or month")

if __name__ == '__main__':
    print("\n🚀 SkyIQ Flight Data Seeder")
    print("This will populate weekly_flight_record with realistic data\n")
    seed_flight_data()