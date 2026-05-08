import sqlite3
import requests
import json, jsonlines
import pandas as pd
import time
import os
import re
import unicodedata
import lxml
from bs4 import BeautifulSoup, SoupStrainer, UnicodeDammit
from pathlib import Path
from datetime import datetime
from html import unescape

from text_processing import normalize_article_text
from utils import DATABASE, ARTICLE_STORAGE, PARSED_TEXT_STORAGE

# DATABASE = 'data/database.db'
# ARTICLE_STORAGE = 'data/articles'
# TEXT_STORAGE = 'data/texts'


def build_id_dict(cursor: sqlite3.Cursor) -> None:
    id_dict = {}
    cursor.execute("""SELECT DISTINCT player_id
                      FROM nf_articles_players""")
    id_list = cursor.fetchall()
    for player_id in id_list:
        player_id = player_id[0]
        # print(player_id)
        cursor.execute("""SELECT person_id FROM sup_people WHERE key_bbref IS ? OR person_id IS ?""", (player_id, player_id))
        result = cursor.fetchone()
        new_id = player_id
        if result is not None:
            new_id = result[0]
        id_dict[player_id] = new_id
    with open(Path(__file__).parent / 'data/id_dict.json', 'w') as f:
        json.dump(id_dict, f, indent=4)


def build_article_map(conn: sqlite3.Connection) -> None:
    with open(Path(__file__).parent / 'data/id_dict.json', 'r') as f:
        id_dict = json.load(f)
    queries = ['article_id >= 1 AND article_id < 1000000', 'article_id >= 1000000 AND article_id < 2000000',
               'article_id >= 2000000 AND article_id < 3000000','article_id >= 3000000 AND article_id < 4000000',
               'article_id >= 4000000 AND article_id < 5000000','article_id >= 5000000 AND article_id < 6000000',
               'article_id >= 6000000 AND article_id < 7000000', 'article_id >= 7000000 AND article_id < 8000000',
               'article_id >= 8000000 AND article_id < 9000000', 'article_id >= 9000000 AND article_id < 10000000',
               'article_id >= 10000000'] # I run out of memory if I do it all at once lol
    sql_base = """SELECT * FROM nf_articles_players WHERE """

    for q in queries:
        print(f'Current query: {q}\n')
        df = pd.read_sql_query(sql_base + q, conn)
        df['person_id'] = df['player_id'].map(id_dict)
        df = df.drop(columns=['player_id']).drop_duplicates()
        df.to_sql(name='new_articles_players', con=conn, if_exists='append', index=False)
        print(df)


def build_player_data_map(conn: sqlite3.Connection) -> None:
    sql = """SELECT person_id, COUNT(person_id) \
              FROM new_articles_players
              GROUP BY person_id"""
    df = pd.read_sql_query(sql, conn)

    sql2 = """SELECT person_id, year_id \
              FROM sup_player_team_seasons
              WHERE (person_id, year_id) IN
                    (SELECT person_id, MIN(year_id) FROM sup_player_team_seasons GROUP BY person_id)
                AND person_id IN (SELECT DISTINCT person_id FROM new_articles_players)"""
    df2 = pd.read_sql_query(sql2, conn)

    sql3 = """SELECT person_id, year_id \
              FROM sup_player_team_seasons
              WHERE (person_id, year_id) IN
                    (SELECT person_id, MAX(year_id) FROM sup_player_team_seasons GROUP BY person_id)
                AND person_id IN (SELECT DISTINCT person_id FROM new_articles_players)"""
    df3 = pd.read_sql_query(sql3, conn)

    df = df.join(df2.set_index('person_id'), on='person_id').drop_duplicates()
    df = df.rename(columns={'COUNT(person_id)': 'article_count','year_id': 'first_season'})
    df = df.join(df3.set_index('person_id'), on='person_id').drop_duplicates()
    df = df.rename(columns={'year_id': 'latest_season'})
    df.to_sql(name='new_player_data', con=conn, if_exists='replace', index=False)
    print(df)


def build_tables_and_views(cursor: sqlite3.Cursor, scratch: bool=False) -> None:
    cursor.execute("""SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='request_data'""")
    table_exists = cursor.fetchone()[0]
    if table_exists:
        print('Table request_data already exists.')
    else:
        cursor.execute("""CREATE TABLE IF NOT EXISTS request_data (
                            'article_id',
                            'article_url',
                            'date_requested',
                            'download_status',
                            'filename',
                            'headers',
                            'exceptions',
                            PRIMARY KEY ('article_id') ON CONFLICT REPLACE
                            )""")
        print("Table request_data created.")

    if scratch: # no scratch option for dropping request_data because that's too valuable to do accidentally
        cursor.execute("""DROP VIEW IF EXISTS url_view""")
        cursor.execute("""DROP VIEW IF EXISTS player_view""")

    cursor.execute("""SELECT COUNT(*) FROM sqlite_master WHERE type = 'view' AND name = 'url_view'""")
    view_exists = cursor.fetchone()[0]
    if view_exists:
        print("View url_view already exists.")
    else:
        # filter URLs to last 10 years of articles from active sites
        cursor.execute("""CREATE VIEW IF NOT EXISTS url_view AS SELECT article_id, article_url FROM nf_articles
                              WHERE site_id IN (SELECT site_id FROM nf_articles_sites WHERE last_post >= '2025-12-01')
                              AND date_inserted >= '2016-01-01 00:00:00'""")
        print("View url_view created.")

    cursor.execute("""SELECT COUNT(*) FROM sqlite_master WHERE type = 'view' AND name = 'player_view'""")
    view_exists = cursor.fetchone()[0]
    if view_exists:
        print("View player_view already exists.")
    else:
        # filter players to those active in 2025 with at least 1000 articles linked to them
        cursor.execute("""CREATE VIEW IF NOT EXISTS player_view AS SELECT * FROM new_player_data
                              WHERE latest_season IS 2025 AND article_count >= 1000""")
        print("View player_view created.")

    cursor.execute("""SELECT COUNT(*)
                      FROM sqlite_master
                      WHERE type = 'view'
                        AND name = 'parsed_article_player_view'""")
    view_exists = cursor.fetchone()[0]
    if view_exists:
        print("View parsed_article_player_view already exists.")
    else:
        cursor.execute("""CREATE VIEW IF NOT EXISTS parsed_article_player_view AS SELECT * FROM new_articles_players
            WHERE article_id IN (SELECT article_id FROM request_data
        )""")
        print("View parsed_article_player_view created.")

    cursor.execute("""SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'updated_player_data'""")
    table_exists = cursor.fetchone()[0]
    if table_exists:
        print('Table updated_player_data already exists.')
    else:
        cursor.execute("""CREATE TABLE IF NOT EXISTS updated_player_data
        (
            'person_id',
            'first_name',
            'last_name',
            'display_name',
            'parsed_article_count',
            'first_season',
            'latest_season',
            PRIMARY KEY ('person_id') ON CONFLICT REPLACE
            )""")
        print("Table updated_player_data created.")
        cursor.execute("""SELECT person_id, COUNT(person_id) FROM parsed_article_player_view GROUP BY person_id""")
        players = cursor.fetchall()[1:]
        for p in players:
            person_id = p[0]
            article_count = p[1]
            cursor.execute("""SELECT person_id, first_season, latest_season FROM new_player_data WHERE person_id LIKE ? AND latest_season IS 2025""", (person_id,))
            seasons = cursor.fetchone()
            if seasons:
                first = seasons[1]
                latest = seasons[2]
                cursor.execute("""SELECT person_id, name_first, name_last, name_display
                                  FROM sup_people
                                  WHERE person_id LIKE ?""", (person_id,))
                name_data = cursor.fetchone()
                cursor.execute("""INSERT INTO updated_player_data
                                  VALUES (?, ?, ?, ?, ?, ?, ?)""",
                               (person_id, name_data[1], name_data[2], name_data[3], article_count, first, latest))
                conn.commit()


def get_content(conn: sqlite3.Connection, cursor: sqlite3.Cursor, article_min_id: int) -> int:
    # select all unique article IDs with person IDs from player_view; get URL from url_view for each article ID
    art_min_id = article_min_id # for running smaller batches to avoid OS errors
    # cursor.execute("""SELECT article_id, article_url FROM url_view WHERE article_id IN
    #                   (SELECT DISTINCT article_id FROM new_articles_players
    #                    WHERE person_id IN (SELECT person_id FROM player_view))
    #                   AND article_id > ?""", (article_min_id,))
    # if working on error correction:
    cursor.execute("""SELECT * FROM request_data WHERE article_url LIKE '%metsmerizedonline.com%' AND article_id > ?""", (article_min_id,))
    articles = cursor.fetchall()
    print(f"{len(articles)} articles identified.")

    # send curl request and store content in file
    for item in articles[:10000]: # for batches
        art_id = item[0]
        if art_id >= 9294711:
            continue
        url = extract_url(item[1])
        # art_name = url.rsplit('/', 1)[1] # specific handling of mlbtraderumors feedproxy urls
        # base = 'https://www.mlbtraderumors.com/'
        # url = base + art_name
        exception = ""
        success = False
        time.sleep(2)  # to avoid connection errors
        # handle exceptions
        try:
            util_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0'}
            r = requests.get(url, timeout=16, headers=util_headers)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            exception = "ConnectionError"
        except requests.exceptions.Timeout:
            exception = "Timeout"
        except requests.exceptions.TooManyRedirects:
            exception = "TooManyRedirects"
        except requests.exceptions.HTTPError:
            exception = "HTTPError"
        except requests.exceptions.RequestException:
            exception = "Other"
        else:
            success = True
            headers = repr(r.headers)
            content = r.text
            # store content in file
            filename = f"{ARTICLE_STORAGE}/{art_id}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        finally:
            timestamp = datetime.now().strftime('%m-%d-%Y %X')
            if not success:
                filename = ""
                headers = ""
            # store article ID, URL, timestamp of request, boolean successful download, filename, headers, exception (if any)
            cursor.execute("""INSERT INTO request_data VALUES(?, ?, ?, ?, ?, ?, ?)""",
                           (art_id, url, timestamp, success, filename, headers, exception))
            conn.commit()
            print(f'Article {art_id} requested (success = {success}).')
            art_min_id = art_id

    return art_min_id


def remove_content(conn: sqlite3.Connection, cursor: sqlite3.Cursor, target: str) -> None:
    sql = """SELECT * FROM request_data WHERE article_url LIKE ?"""
    del_req = """DELETE FROM request_data WHERE article_url LIKE ?"""
    if target.isdigit():
        target = int(target)
        sql = """SELECT * FROM request_data WHERE article_id IS ?"""
        del_req = """DELETE FROM request_data WHERE article_id IS ?"""
    df = pd.read_sql_query(sql, conn, params=(target,))
    # pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    print(df)

    cursor.execute(sql, (target,))
    articles = cursor.fetchall()
    # del_count = 0
    # for item in articles:
    #     art_id = item[0]
    #     paths = [f'{ARTICLE_STORAGE}/{art_id}.txt', f'{TEXT_STORAGE}/{art_id}_final.txt']
    #     for file_path in paths:
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #             print(f'File {file_path} removed.')
    #             del_count += 1
    #         else:
    #             print(f'File {file_path} does not exist.')
    # print(f'{del_count} files removed.')

    with jsonlines.open('data/removed_content.jsonl', 'a') as f:
        f.write_all(articles)

    cursor.execute(del_req, (target,))
    conn.commit()


def restore_content(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    cursor.execute("""SELECT article_id, article_url
                      FROM url_view
                      WHERE article_id IN
                            (SELECT DISTINCT article_id
                             FROM new_articles_players
                             WHERE person_id IN (SELECT person_id FROM player_view))
                        AND article_url LIKE '%birdswatcher%'""")
    articles = cursor.fetchall()
    restore_count = 0
    for item in articles:
        art_id = item[0]
        url = item[1]
        filename = f'{ARTICLE_STORAGE}/{art_id}.txt'
        if os.path.exists(filename):
            timestamp = datetime.now().strftime('%m-%d-%Y %X')
            cursor.execute("""INSERT INTO request_data
                              VALUES (?, ?, ?, ?, ?, ?, ?)""",
                           (art_id, url, timestamp, True, filename, '{}', 'Restored from deletion'))
            conn.commit()
            restore_count += 1
    print(f'{restore_count} article records restored.')


def extract_url(text):
    text = text.split('.html/')[0]
    parts = text.rsplit('/', maxsplit=3)
    # print(parts)
    new_url = '/'.join([parts[0], parts[3]])
    # print(new_url)
    return new_url


def parse_html(conn: sqlite3.Connection, cursor: sqlite3.Cursor, target_url: str) -> None:
    cursor.execute("""SELECT * FROM request_data WHERE article_url LIKE ?""", (target_url,))
    articles = cursor.fetchall()
    art_class = 'available-content'
    trash_classes = ['wp-caption-text', 'twitter-tweet', 'twitter-video', 'jp-relatedposts', 'screen-reader-text', 'instagram-media', 'tr-caption', 'sd-title', 'sharedaddy sd-block sd-like jetpack-likes-widget-wrapper jetpack-likes-widget-loaded', 'sharedaddy sd-sharing-enabled', 'wp-socializer wpsr-share-icons', 'mvp-related-posts left relative']
    trash_attrs = {'id': ['categoryList']}
    text_tags = ['p', 'h3', 'h2', 'h4', 'blockquote']

    for item in articles:
        art_id = item[0]
        file_path = f"{ARTICLE_STORAGE}/{art_id}.txt"
        html_tags = SoupStrainer(name='div', class_=art_class)
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, features='lxml', from_encoding='utf-8', parse_only=html_tags)
        br_tags = soup.find_all('br')
        replaced_tags = [t.replace_with('\n') for t in br_tags]
        trash_tags = soup.find_all(class_=trash_classes) + soup.find_all(attrs=trash_attrs)
        for tag in trash_tags:
            tag.clear()
        body = soup.find_all(text_tags) # change as needed
        text = '\n'.join([tag.text for tag in body])
        # text = soup.text
        text = normalize_article_text(text)
        # print(text)
        out_path = f"{PARSED_TEXT_STORAGE}/{art_id}_final.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        timestamp = datetime.now().strftime('%m-%d-%Y %X')
        cursor.execute("""INSERT INTO request_data
                          VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (art_id, item[1], timestamp, 2, out_path, item[5], item[6]))
        conn.commit()
        print(f'Article {art_id} text written to file.')
    print(f'{len(articles)} articles written to file.')


def remove_small_texts(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    # init list for info
    target_texts = []
    # get ALL articles from table
    sql = """SELECT * FROM request_data"""
    # df = pd.read_sql_query(sql, conn)
    # print(df)
    cursor.execute(sql)
    articles = cursor.fetchall()
    for item in articles:
        art_id = item[0]
        file_path = f'{PARSED_TEXT_STORAGE}/{art_id}_final.txt'
        if os.path.exists(file_path):
            filesize = os.path.getsize(file_path)
            if filesize < 400:
                target_texts.append(item)
        else:
            print(f'File {file_path} does not exist.')

    # TODO: change to jsonlines
    # write list to jsonlines file
    with open(Path(__file__).parent / 'data/small_texts_1.json', 'w') as f:
        json.dump(target_texts, f, indent=4)
    for item in target_texts:
        art_id = str(item[0])
        remove_content(conn, cursor, art_id)


def update_player_article_counts(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    cursor.execute("""DROP VIEW IF EXISTS parsed_article_player_view""")
    cursor.execute("""CREATE VIEW IF NOT EXISTS parsed_article_player_view AS SELECT * FROM new_articles_players
        WHERE article_id IN (SELECT article_id FROM request_data)""")
    print("View parsed_article_player_view updated.")

    cursor.execute("""DROP VIEW IF EXISTS subset_article_player_view""")
    cursor.execute("""CREATE VIEW IF NOT EXISTS subset_article_player_view AS SELECT * FROM new_articles_players
        WHERE article_id IN (SELECT article_id FROM data_subset
    )""")
    print("View subset_article_player_view updated.")

    # cursor.execute("""SELECT person_id, COUNT(person_id)
    #                   FROM subset_article_player_view
    #                   WHERE person_id IN (SELECT person_id FROM updated_player_data)
    #                   GROUP BY person_id""")
    # players = cursor.fetchall()[1:]
    # for p in players:
    #     person_id = p[0]
    #     article_count = p[1]
    #     cursor.execute("""SELECT *
    #                       FROM updated_player_data
    #                       WHERE person_id LIKE ?""", (person_id,))
    #     entry = cursor.fetchone()
    #     cursor.execute("""INSERT INTO updated_player_data
    #                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
    #                    (person_id, entry[1], entry[2], entry[3], article_count, entry[5], entry[6], 0))
    #     conn.commit()
    # print("Player article counts updated.")

    # create updated_player_view for updated_player_data w/ min article count
    cursor.execute("""DROP VIEW IF EXISTS updated_player_view""")
    cursor.execute("""CREATE VIEW IF NOT EXISTS updated_player_view AS
    SELECT *
    FROM updated_player_data
    WHERE parsed_article_count >= 1000""")
    print("View updated_player_view updated.")


# if __name__ == '__main__':
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()

    # update_player_article_counts(conn, cursor)

    # remove_small_texts(conn, cursor)

    # remove_content(conn, cursor, '%lastwordonsports%')
    # art_ids = ['9671416',]
    # for site in sites:
    #     remove_content(conn, cursor, f'%{site}%')

    # restore_content(conn, cursor)

    # for site in sites:
    #     parse_html(conn, cursor, f'%{site}%')

    # building necessary data views:
    # build_id_dict(cursor)
    # build_article_map(conn)
    # build_player_data_map(conn)
    # build_tables_and_views(cursor, scratch=False)

    # article_min_id = 3243692 # change as needed
    # for _ in range(2): # change range as needed for number of articles
    #     article_min_id = get_content(conn, cursor, article_min_id)