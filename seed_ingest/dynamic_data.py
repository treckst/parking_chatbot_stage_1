"""Create fictional November 2026 parking fixtures once."""

import sqlite3
from datetime import date
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "parking.db"

# Simplified PLN/hour rates, ignoring free periods and daily caps.
PARKINGS = [
    ("A", "Łódź, al. Piłsudskiego 15/23", "Galeria Łódzka", 5.0),
    ("B", "Łódź, Drewnowska 58", "Manufaktura", 0.0),
    ("C", "Łódź, Pabianicka 245", "Port Łódź", 0.0),
    ("D", "Łódź, al. Politechniki 1", "Nowa Sukcesja", 6.0),
]
POSITIONS = ("row A, near entrance", "row A, middle", "row A, end of row")

SCHEMA = """
CREATE TABLE parking_slots (
    id INTEGER PRIMARY KEY,
    slot_code TEXT NOT NULL,
    parking_date TEXT NOT NULL,
    location TEXT NOT NULL,
    parking_name TEXT NOT NULL,
    spot_details TEXT NOT NULL,
    price_per_hour_pln REAL NOT NULL,
    opens_at TEXT NOT NULL,
    closes_at TEXT NOT NULL,
    is_available INTEGER NOT NULL,
    UNIQUE (slot_code, parking_date)
)
"""


def seed_november(db):
    slots = [
        (f"{prefix}{number:02d}", location, name, details, price)
        for prefix, location, name, price in PARKINGS
        for number, details in enumerate(POSITIONS, 1)
    ]

    for day in range(1, 31):
        parking_day = date(2026, 11, day)

        # Fictional opening hours.
        if parking_day.weekday() >= 5:
            opens_at, closes_at = "08:00", "20:00"
        else:
            opens_at, closes_at = "06:00", "22:00"

        for index, (code, location, name, details, price) in enumerate(slots):
            available = int((day + index) % 11 != 0)

            db.execute("""
                INSERT INTO parking_slots (
                    slot_code, parking_date, location, parking_name,
                    spot_details, price_per_hour_pln,
                    opens_at, closes_at, is_available
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code, parking_day.isoformat(), location, name,
                details, price, opens_at, closes_at, available,
            ))


def create_mock_database(database_path=DATABASE_PATH):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=False)

    try:
        db = sqlite3.connect(path)
        try:
            db.execute(SCHEMA)
            seed_november(db)
            db.commit()
        finally:
            db.close()
    except Exception:
        path.unlink()
        raise

    print(f"Created {path}: 360 daily parking slots.")


if __name__ == "__main__":
    create_mock_database()