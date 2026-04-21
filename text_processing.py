import sqlite3
import json, jsonlines
from collections import Counter, defaultdict
import pandas as pd
import time
import os
import re
import unicodedata
import random
import warnings
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from html import unescape
from gensim.utils import tokenize, deaccent


DATABASE = 'data/database.db'
ARTICLE_STORAGE = 'data/articles'
PARSED_TEXT_STORAGE = 'data/texts/parsed'
CLEAN_TEXT_STORAGE = 'data/texts/clean'
TKN_TEXT_STORAGE = 'data/texts/tokenized'
MENT_TEXT_STORAGE = 'data/texts/mention_replaced'


def preprocess_tokens(cursor):
    cursor.execute("""SELECT * FROM request_data""")
    texts = cursor.fetchall()

    print(f'Tokenizing {len(texts)} texts.')
    for item in tqdm(texts):
        art_id = item[0]
        orig_path = item[4]
        out_path = f'{TKN_TEXT_STORAGE}/{art_id}_tkn.txt'
        if os.path.exists(orig_path):
            with open(orig_path, 'r', encoding='utf-8') as f:
                txt = f.read()
            tkn_txt = ' '.join(tokenize(txt, lowercase=True, deacc=True))
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(tkn_txt)
            # print(f'Article {art_id} tokenized.')
        else:
            warnings.warn(message=f'File path {orig_path} for article {art_id} does not exist.')


def normalize_article_text(text):
    text = unescape(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub('(Â)|(&amp;nbsp;)|(&nbsp;)|( )', '', text)
    text = re.sub('(â)|(â)|(â€TM)|(â€ ̃)', "'", text)
    text = re.sub('(â¦)|(â€¦)', "...", text)
    text = re.sub('(â)|(â)|(â€œ)|(â€)|(â3)', '"', text)
    text = re.sub('(â)|(â€“)|(â€”)', '—', text)
    text = re.sub('â', '–', text)
    text = re.sub('Ã3', 'ó', text)
    text = re.sub('Ã©', 'é', text)
    text = re.sub('Ã­', 'í', text)
    text = re.sub('Ão', 'ú', text)
    text = re.sub('Ã¡', 'á', text)
    text = re.sub('Ã', 'Á', text)
    text = re.sub('Ã±', 'ñ', text)
    text = re.sub('Ã1⁄4', 'ü', text)
    text = re.sub('( )|(​)|(\t)|( {2,})', ' ', text)
    return text


def replace_mentions(conn, cursor):
    '''
    6. search for mentions in entire file w/ regex
        1. full name
        3. last name only (w/ lookbehind)
    7. after article is finished, count up all mentions and add to 'total' field
    8. return dict
    '''
    with jsonlines.open('data/regex_dict.jsonl') as f:
        regex_dict = {obj['id']: obj['regex'] for obj in f}
    article_mention_tracker = Counter()
    player_mention_tracker = {}
    cursor.execute("""SELECT article_id FROM request_data""")
    arts = cursor.fetchall()
    article_player_map = {a[0]: [] for a in arts}

    # player_mention_tracker = {
    #     'verlan001jus': {2932231: 0, 2932233: 0},
    #     'altuve001jos': {2932233: 0}
    # }
    # article_player_map = {
    #     2932231: ['verlan001jus'],
    #     2932233: ['verlan001jus', 'altuve001jos'],
    # }

    sql = """SELECT article_id FROM parsed_article_player_view WHERE person_id LIKE ?"""
    for player in tqdm(regex_dict):
        cursor.execute(sql, (player,))
        articles = [x[0] for x in cursor.fetchall()]
        player_mention_tracker[player] = {art: 0 for art in articles}
        for art in articles:
            article_player_map[art].append(player)

    for art_id in tqdm(article_player_map):
        filepath = f'{TKN_TEXT_STORAGE}/{art_id}_tkn.txt'
        with open(filepath, 'r', encoding='utf-8') as f:
            art = f.read()
        new_art = art
        for p in article_player_map[art_id]:
            names = regex_dict[p]
            ptn = f"{names['full']}"
            (new_art, ments) = re.subn(ptn, p, new_art, flags=re.I)
            if ments == 0:
                continue
            player_mention_tracker[p][art_id] += ments
            article_mention_tracker[art_id] += ments
            ptn = rf"(?<!{names['first']} ){names['last']}(?!\S)"
            if 'alt' in names:
                ptn = rf"(?:(?<!{names['first']} )(?<!{names['alt']} )){names['last']}(?!\S)"
            (new_art, ments) = re.subn(ptn, p, new_art, flags=re.I)
            player_mention_tracker[p][art_id] += ments
            article_mention_tracker[art_id] += ments
        out_path = f'{MENT_TEXT_STORAGE}/{art_id}_ment.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_art)

    with open('data/exp_001/player_ments_001.json', 'w', encoding='utf-8') as f:
        json.dump(player_mention_tracker, f, ensure_ascii=False, indent=4)
    with open('data/exp_001/article_ments_001.json', 'w', encoding='utf-8') as f:
        json.dump(article_mention_tracker, f, ensure_ascii=False, indent=4)


def clean_text(conn, cursor, target_url):
    # get all articles from request data
    cursor.execute("""SELECT *
                      FROM request_data
                      WHERE article_url LIKE ?""", (target_url,))
    articles = cursor.fetchall()
    random.shuffle(articles)

    line_flags = [
        '^Thanks for reading',
        '^http',
        '^.*mp4$',
        '^Next.*:',
        '^More from',
        '^Related.*:',
        '^Embed from',
        '^Continue reading',
        '^Featured (?:photo|image)',
        '^Read.*:',
        '^See also',
        '^More.*:$', # good general point
        '^Want your',
        '^Write for',
        '^Notes:?$',
        # r'^\*+.+\*+$',
        '^Betsided',
        '^Notes from around baseball',
        r'^MiLB\.com image',
        '^Draft Prep',
        '^Uh-Oh!:',
        # '^{subscribe',
        # 'Check out our',
        # '^Check out',
        # '^Check.+for more Mets',
        # '^.+to check out',
        # '^Enjoy!$',
        # '^Provided by',
        # '^Generated'
        # 'wpedon',
        # r'^\[',
        # r'^(\S+ )?Courtesy of',
        # 'Photo (courtesy|by)',
        # 'More.*news.*:$',
        # '^Interested in learning',
        # '^Please help support',
        # '^Discuss',
        '^Box *score',
        # 'Click here',
        '^Answer this',
        '^View results',
        # r'^By \S+ \S+$',
    ]
    end_flags = [
        # '^Houston Astros News and Notes',
        # '^Phillies Game Today',
        # '^Also Read',
        # '^Access weekly',
        # 'Poll (link|below)',
        # r'^(\S+ )?podcast episodes',
        # "^If you're a Caretaker",
        # '^Main (?:photo|image)',
        # '^Tags$',
        # '^Subscribe to',
        # '^Beer and baseball',
        # '^Photo credit',
        # '^Players mentioned',
        # '^More from Phillies Nation',
        # '^Ticket IQ',
    ]

    for item in articles:
        art_id = item[0]
        file_path = f'{PARSED_TEXT_STORAGE}/{art_id}_final.txt'
        clean_lines = []
        end_flag = False
        with open(file_path, 'r', encoding='utf-8') as f:
            # read in line by line
            for line in f:
                line_flag = False
                line = line.strip() # if empty, continue
                if line:
                    # normalize text first
                    line = normalize_article_text(line)
                    line = line.strip() # just in case of any weirdness
                    if not line:
                        continue
                    for flag in line_flags:
                        if re.search(flag, line, re.I) is not None:
                            line_flag = True
                            break
                    if line_flag: # if line flag at start of/in line, skip
                        continue
                    for flag in end_flags:
                        if re.search(flag, line, re.I) is not None:
                            end_flag = True
                            break
                    if end_flag: # if end flag in line, end loop entirely
                        break
                    # if clean, append to list
                    clean_lines.append(line)
        cleaned_text = '\n'.join(clean_lines) # for readability

        new_file_path = f'{CLEAN_TEXT_STORAGE}/{art_id}_clean.txt'
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        timestamp = datetime.now().strftime('%m-%d-%Y %X')
        cursor.execute("""INSERT INTO request_data
                          VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (art_id, item[1], timestamp, 3, new_file_path, item[5], item[6]))
        conn.commit()
        print(f'Article {art_id} text cleaned.')

    print(f'{len(articles)} articles cleaned.')


def build_regex_dict(conn, cursor, update=False):
    dict_path = 'data/regex_dict.jsonl'
    mode = 'w'
    if update:
        if os.path.exists(dict_path):
            with jsonlines.open(dict_path, 'r', encoding='utf-8') as f:
                id_list = [obj['id'] for obj in f]
            mode = 'a'
        else:
            raise OSError(f'No existing {dict_path} file.')

    with open('data/player_list.txt', 'r', encoding='utf-8') as f:
        player_list = [line.strip() for line in f]
    players = []
    excluded = []
    for p in player_list:
        cursor.execute("""SELECT person_id, display_name
                          FROM updated_player_view WHERE person_id LIKE ?""", (p,))
        player_info = cursor.fetchone()
        if player_info:
            players.append(player_info)
        else:
            excluded.append(p)

    # cursor.execute("""SELECT person_id, display_name
    #                   FROM updated_player_view""")
    # players = cursor.fetchall()

    if update:
        name_dict_list = [{'id': p[0], 'regex': {'full': deaccent(p[1])}} for p in players if p[0] not in id_list]
    else:
        name_dict_list = [{'id': p[0], 'regex': {'full': deaccent(p[1])}} for p in players]
    for p in name_dict_list:
        name_parts = p['regex']['full'].split()
        p['regex']['first'] = name_parts[0]
        p['regex']['last'] = name_parts[1]
        # if len(name_parts) > 2:
        #     p['regex']['extra'] = name_parts[2]
    with jsonlines.open(dict_path, mode=mode) as f:
        f.write_all(name_dict_list)


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    replace_mentions(conn, cursor)

    # preprocess_tokens(cursor)

    # site = 'housethathankbuilt'
    # clean_text(conn, cursor, f'%{site}%')

    # build_regex_dict(conn, cursor, update=True)