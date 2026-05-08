import os
from contextlib import contextmanager
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / 'data'
# DATABASE = f'{DATA_DIR}/database.db'
DATABASE = f'{DATA_DIR}/war_db.db'
ARTICLE_STORAGE = f'{DATA_DIR}/articles'
PARSED_TEXT_STORAGE = f'{DATA_DIR}/texts/parsed'
CLEAN_TEXT_STORAGE = f'{DATA_DIR}/texts/clean'
TKN_TEXT_STORAGE = f'{DATA_DIR}/texts/tokenized'
MASK_TEXT_STORAGE = f'{DATA_DIR}/texts/masked'
MENT_TEXT_STORAGE = f'{DATA_DIR}/texts/mention_replaced'
RANDOM_SEED = 42

print(f'{DATA_DIR}, {MASK_TEXT_STORAGE}, {MENT_TEXT_STORAGE}')


@contextmanager
def ch_dir(new_path):
    old_path = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(old_path)