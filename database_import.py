import sqlite3
from pathlib import Path

DATABASE = Path(__file__).parent / 'data/database.db'
# DATA_SOURCE = Path(__file__).parent / 'data/newsfeed-dial-2025-12-29.sql'
DATA_SOURCE = Path(__file__).parent / 'data/update_schema.sql'

if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    with open(DATA_SOURCE, 'r', encoding='utf8') as f:
        cursor.executescript(f.read())