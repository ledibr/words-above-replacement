import sqlite3, json, jsonlines, time, os, re, unicodedata, random, warnings
import pandas as pd
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from html import unescape
from gensim.utils import tokenize, deaccent
from utils import (DATA_DIR, DATABASE, ARTICLE_STORAGE, PARSED_TEXT_STORAGE, CLEAN_TEXT_STORAGE,
                   TKN_TEXT_STORAGE, MASK_TEXT_STORAGE, MENT_TEXT_STORAGE, RANDOM_SEED, ch_dir)


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


def replace_mentions(cursor):
    cursor.execute("""SELECT COUNT(*)
                      FROM sqlite_master
                      WHERE type = 'view'
                        AND name = 'data_subset'""")
    view_exists = cursor.fetchone()[0]
    if view_exists:
        print("View data_subset already exists.")
    else:
        cursor.execute("""CREATE VIEW IF NOT EXISTS data_subset AS
        SELECT *
        FROM request_data
        WHERE article_url NOT LIKE '%metsmerizedonline%'""")
        print('View data_subset created.')

    with jsonlines.open(f'regex_dict.jsonl') as f:
        regex_dict = {obj['id']: obj['regex'] for obj in f}
    article_mention_tracker = Counter()
    player_mention_tracker = {}
    cursor.execute("""SELECT article_id FROM data_subset""")
    arts = cursor.fetchall()
    article_player_map = {a[0]: [] for a in arts}

    sql = """SELECT article_id FROM parsed_article_player_view WHERE person_id LIKE ? AND article_id IN (SELECT article_id FROM data_subset)"""
    for player in tqdm(regex_dict):
        cursor.execute(sql, (player,))
        articles = [x[0] for x in cursor.fetchall()]
        player_mention_tracker[player] = Counter()
        for art in articles:
            article_player_map[art].append(player)

    for art_id in tqdm(article_player_map):
        # filepath = f'{TKN_TEXT_STORAGE}/{art_id}_tkn.txt'
        filepath = f'{MASK_TEXT_STORAGE}/{art_id}_mask.txt'
        with open(filepath, 'r', encoding='utf-8') as f:
            art = f.read()
        new_art = art
        for p in regex_dict:
            names = regex_dict[p]
            ptn = rf"{names['full']}"
            new_art, ments = re.subn(ptn, p, new_art, flags=re.I)
            if ments == 0 and p not in article_player_map[art_id]:
                continue
            elif ments > 0 and p not in article_player_map[art_id]:
                article_player_map[art_id].append(p)
            player_mention_tracker[p][art_id] += ments
            article_mention_tracker[art_id] += ments
            ptn = rf"{names['last']}(?!\S)"
            new_art, ments = re.subn(ptn, p, new_art, flags=re.I)
            player_mention_tracker[p][art_id] += ments
            article_mention_tracker[art_id] += ments
        out_path = f'{MENT_TEXT_STORAGE}/special/{art_id}_ment.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_art)

    os.makedirs('mentions', exist_ok=True)
    with open(f'mentions/player_mentions.json', 'w', encoding='utf-8') as f:
        json.dump(player_mention_tracker, f, ensure_ascii=False, indent=4)
    with open(f'mentions/article_mentions.json', 'w', encoding='utf-8') as f:
        json.dump(article_mention_tracker, f, ensure_ascii=False, indent=4)
    with open(f'mentions/article_players.json', 'w', encoding='utf-8') as f:
        json.dump(article_player_map, f, ensure_ascii=False, indent=4)


def mask_mentions(conn, cursor):
    cursor.execute("""SELECT article_id
                      FROM request_data""")
    texts = cursor.fetchall()

    with open('data/team_names.json', 'r', encoding='utf-8') as f:
        teams = json.load(f)
    locs = list(set([team['loc'] for team in teams]))
    ids = [team['id'] for team in teams] + ['Red', 'Sox']
    abbs = [team['abb'] for team in teams]
    leagues = ['american league', 'al', 'national league', 'nl']

    for item in tqdm(texts):
        art_id = item[0]
        orig_path = f'{TKN_TEXT_STORAGE}/{art_id}_tkn.txt'
        out_path = f'{MASK_TEXT_STORAGE}/{art_id}_mask.txt'
        if os.path.exists(orig_path):
            with open(orig_path, 'r', encoding='utf-8') as f:
                mask_txt = f.read()
            for x in locs:
                mask_txt = re.sub(rf'\b{x}\b', '##LOC', mask_txt, flags=re.I)
            for x in ids:
                mask_txt = re.sub(rf'\b{x}\b', '##TEAM', mask_txt, flags=re.I)
            for x in abbs:
                mask_txt = re.sub(rf'\b{x}\b', '##TEAM', mask_txt, flags=re.I)
            for x in leagues:
                mask_txt = re.sub(rf'\b{x}\b', '##LEAGUE', mask_txt, flags=re.I)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(mask_txt)
        else:
            warnings.warn(message=f'File path {orig_path} for article {art_id} does not exist.')


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


def build_regex_dict(cursor, update=False):
    dict_path = 'regex_dict.jsonl'
    mode = 'w'
    if update:
        if os.path.exists(dict_path):
            with jsonlines.open(dict_path, 'r', encoding='utf-8') as f:
                id_list = [obj['id'] for obj in f]
            mode = 'a'
        else:
            raise OSError(f'No existing {dict_path} file.')

    with open('player_list.txt', 'r', encoding='utf-8') as f:
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
    print(f'Excluded players: {", ".join([x for x in excluded])}')

    if os.path.exists(dict_path) and not update:
        print(f'Player regex dictionaries location: {os.getcwd()}/{dict_path}')
    elif not os.path.exists(dict_path):
        if update:
            name_dict_list = [{'id': p[0], 'regex': {'full': deaccent(p[1])}} for p in players if p[0] not in id_list]
        else:
            name_dict_list = [{'id': p[0], 'regex': {'full': deaccent(p[1])}} for p in players]
        for p in name_dict_list:
            name_parts = p['regex']['full'].split(maxsplit=1)
            p['regex']['first'] = name_parts[0]
            p['regex']['last'] = name_parts[1]
            # if len(name_parts) > 2:
            #     p['regex']['extra'] = name_parts[2]
        with jsonlines.open(dict_path, mode=mode) as f:
            f.write_all(name_dict_list)
        print(f'Player regex dictionaries written to {os.getcwd()}/{dict_path}.')


def get_player_list(cursor):
    cursor.execute("""SELECT person_id FROM updated_player_view""")
    ids = cursor.fetchall()
    id_lines = '\n'.join([x[0] for x in ids])
    fname = 'player_list.txt'
    if os.path.exists(fname):
        print(f'Player list location: {os.getcwd()}/{fname}')
    else:
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(id_lines)
        print(f'Player list written to {os.getcwd()}/{fname}.')


def update_mention_counts(conn, cursor):
    cursor.execute("""CREATE TABLE IF NOT EXISTS main_w2v_player_data AS SELECT * FROM updated_player_view""")

    with open('mentions/player_mentions.json', 'r', encoding='utf-8') as f:
        player_mentions = json.load(f)
    mention_counts = {p: sum(player_mentions[p].values()) for p in player_mentions}
    article_counts = {p: len([a for a in player_mentions[p] if player_mentions[p][a] > 0]) for p in player_mentions}
    for p in player_mentions:
        cursor.execute("""UPDATE main_w2v_player_data SET parsed_article_count = ?, mention_count = ? WHERE person_id LIKE ?""", (article_counts[p], mention_counts[p], p))
        conn.commit()


# if __name__ == '__main__':
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()

    # exp_num = '005'
    # exp_path = f'data/exp_{exp_num}'
    # os.makedirs(exp_path, exist_ok=True)

    # with ch_dir(exp_path):
    #     # get_player_list(cursor)
    #     # build_regex_dict(cursor)
    #     # replace_mentions(cursor)
    #     update_mention_counts(conn, cursor)

    # mask_mentions(conn, cursor)

    # preprocess_tokens(cursor)

    # site = 'housethathankbuilt'
    # clean_text(conn, cursor, f'%{site}%')