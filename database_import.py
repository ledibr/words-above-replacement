import sqlite3, bz2
from utils import DATABASE, DATA_DIR

# DATABASE = Path(__file__).parent / 'data/database.db'
DATA_SOURCE = DATA_DIR / 'war_db.db.txt.bz2'

if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    with bz2.open(DATA_SOURCE, 'rt', encoding='utf-16') as f:
        cursor.executescript(f.read())