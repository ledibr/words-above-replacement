import json, os, random, sqlite3, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from adjustText import adjust_text
from gensim.models import Word2Vec
from collections import Counter
from scipy.stats import linregress
from tqdm import tqdm
from utils import DATA_DIR, DATABASE, ARTICLE_STORAGE, PARSED_TEXT_STORAGE, CLEAN_TEXT_STORAGE, TKN_TEXT_STORAGE, MASK_TEXT_STORAGE, MENT_TEXT_STORAGE, RANDOM_SEED, ch_dir


# future stats of interest: DRS, OAA, sprint speed, CSW%, avg. velo

def sort_players(cursor):
    with open(glob.glob('player_list*')[0], 'r', encoding='utf-8') as f:
        player_ids = [line.strip() for line in f]
    pos_tracker = {p: {} for p in player_ids}

    sql = """SELECT person_id, primary_pos_season, team_id FROM sup_player_team_seasons WHERE person_id LIKE ? AND year_id >= 2016 AND phase_id IS 'reg'"""
    for p in tqdm(player_ids):
        cursor.execute(sql, (p,))
        res = cursor.fetchall()
        pos_tracker[p]['pos'] = Counter([item[1] for item in res]).most_common(1)[0][0]
        pos_tracker[p]['team'] = Counter([item[2] for item in res]).most_common(1)[0][0]

    return pos_tracker


def get_pitcher_stats(cursor, players):
    psr = glob.glob('stats/pitcher_stats_raw.csv')
    if len(psr) > 0:
        return pd.read_csv(psr[0], encoding='utf-8')

    sql = """SELECT person_id, year_id, p_g, p_gs, p_er, p_ip_outs, p_bb, p_h, p_so, p_war, p_game_score_sum, p_qs 
             FROM sup_player_team_seasons WHERE person_id LIKE ? AND year_id >= 2016 AND phase_id IS 'reg'"""
    raw_stats = {p: {} for p in players}
    stat_list = ['p_g', 'p_gs', 'p_er', 'p_ip_outs', 'p_bb', 'p_h', 'p_so', 'p_war', 'p_game_score_sum', 'p_qs']
    for p in tqdm(players):
        cursor.execute(sql, (p,))
        res = cursor.fetchall()
        game_sum = sum([float(item[2]) if item[2] is not None else 0 for item in res])
        start_sum = sum([float(item[3]) if item[3] is not None else 0 for item in res])
        if (start_sum / game_sum) < 0.6:
            raw_stats[p]['position'] = 'RP'
        else:
            raw_stats[p]['position'] = 'SP'
        raw_stats[p]['years'] = len(set([i[1] for i in res]))
        raw_stats[p]['team'] = players[p]['team']
        for i in range(len(stat_list)):
            raw_stats[p][stat_list[i]] = sum([float(item[i + 2]) if item[i + 2] is not None else 0.0 for item in res])

    df = pd.DataFrame.from_dict(raw_stats, orient='index')
    df = df.rename_axis('person_id').reset_index()
    df.to_csv('stats/pitcher_stats_raw.csv', index=False, mode='w', encoding='utf-8', float_format='%.4f')

    return df


def get_batter_stats(cursor, players):
    bsr = glob.glob('stats/batter_stats_raw.csv')
    if len(bsr) > 0:
        return pd.read_csv(bsr[0], encoding='utf-8')

    sql = """SELECT person_id, year_id, b_games, b_ab, b_pa, b_h, b_bb, b_hbp, b_sf, b_tb, b_so, \
                    b_hr, b_war, b_wpa_bat, b_leverage_index_avg, b_wpa_li_adjusted
             FROM sup_player_team_seasons WHERE person_id LIKE ? AND year_id >= 2016 AND phase_id IS 'reg'"""
    raw_stats = {p: {} for p in players}
    stat_list = ['b_games', 'b_ab', 'b_pa', 'b_h', 'b_bb', 'b_hbp', 'b_sf', 'b_tb', 'b_so', 'b_hr',
                 'b_war', 'b_wpa_bat', 'b_leverage_index_avg', 'b_wpa_li_adjusted']
    for p in tqdm(players):
        cursor.execute(sql, (p,))
        res = cursor.fetchall()
        raw_stats[p]['position'] = players[p]['pos']
        raw_stats[p]['years'] = len(set([i[1] for i in res]))
        raw_stats[p]['team'] = players[p]['team']
        for i in range(len(stat_list)):
            raw_stats[p][stat_list[i]] = sum([float(item[i + 2]) for item in res if item[i + 2] is not None])
            if stat_list[i] == 'b_leverage_index_avg':
                raw_stats[p][stat_list[i]] = raw_stats[p][stat_list[i]] / len(res)
        raw_stats[p]['b_clutch'] = sum([(item[13] / item[14]) - item[15] for item in res if item[13] is not None and item[14] is not None and item[15] is not None])

    df = pd.DataFrame.from_dict(raw_stats, orient='index')
    df = df.rename_axis('person_id').reset_index()
    df.to_csv('stats/batter_stats_raw.csv', index=False, mode='w', encoding='utf-8', float_format='%.4f')

    return df


def calc_pitcher_stats(df):
    psc = glob.glob('stats/pitcher_stats_calc.csv')
    if len(psc) > 0:
        return pd.read_csv(psc[0], encoding='utf-8')

    calc_stats = {p.person_id: {'position': p.position, 'years': p.years, 'team': p.team} for p in df.itertuples(index=False)}
    for p in df.itertuples(index=False):
        p_id = p.person_id
        p_ip = p.p_ip_outs / 3
        calc_stats[p_id]['p_era'] = (p.p_er / p_ip) * 9
        calc_stats[p_id]['p_whip'] = (p.p_bb + p.p_h) / p_ip
        calc_stats[p_id]['p_k9'] = (p.p_so / p_ip) * 9
        calc_stats[p_id]['p_bb9'] = (p.p_bb / p_ip) * 9
        calc_stats[p_id]['p_k_bb'] = p.p_so / p.p_bb
        if p.p_gs > 0:
            calc_stats[p_id]['p_avg_game_score'] = p.p_game_score_sum / p.p_gs
            calc_stats[p_id]['p_qs_rate'] = p.p_qs / p.p_gs
        else:
            calc_stats[p_id]['p_avg_game_score'] = 0.0
            calc_stats[p_id]['p_qs_rate'] = 0.0
        calc_stats[p_id]['p_war_162'] = p.p_war * (68 / (p.p_g + p.p_gs))

    df = pd.DataFrame.from_dict(calc_stats, orient='index')
    df = df.rename_axis('person_id').reset_index()
    df.to_csv('stats/pitcher_stats_calc.csv', index=False, mode='w', encoding='utf-8', float_format='%.3f')

    return df


def calc_batter_stats(df):
    bsc = glob.glob('stats/batter_stats_calc.csv')
    if len(bsc) > 0:
        return pd.read_csv(bsc[0], encoding='utf-8')

    calc_stats = {p.person_id: {'position': p.position, 'years': p.years, 'team': p.team} for p in df.itertuples(index=False)}
    for p in df.itertuples(index=False):
        p_id = p.person_id
        calc_stats[p_id]['b_avg'] = p.b_h / p.b_ab
        calc_stats[p_id]['b_obp'] = (p.b_h + p.b_bb + p.b_hbp) / (p.b_ab + p.b_bb + p.b_hbp + p.b_sf)
        calc_stats[p_id]['b_slg'] = p.b_tb / p.b_ab
        calc_stats[p_id]['b_iso'] = (p.b_tb - p.b_h) / p.b_ab
        calc_stats[p_id]['b_ops'] = calc_stats[p_id]['b_obp'] + calc_stats[p_id]['b_slg']
        calc_stats[p_id]['b_k_rate'] = p.b_so / p.b_pa
        calc_stats[p_id]['b_hr_rate'] = p.b_hr / p.b_ab
        calc_stats[p_id]['b_clutch'] = p.b_clutch
        calc_stats[p_id]['b_war_162'] = p.b_war / (p.b_games / 162)

    df = pd.DataFrame.from_dict(calc_stats, orient='index')
    df = df.rename_axis('person_id').reset_index()
    df.to_csv('stats/batter_stats_calc.csv', index=False, mode='w', encoding='utf-8', float_format='%.3f')

    return df


def get_stats(cursor, embeds):
    players = sort_players(cursor)
    pitchers = {k: v for k, v in players.items() if v['pos'] == 'P'}
    batters = {k: v for k, v in players.items() if v['pos'] != 'P'}

    os.makedirs('stats', exist_ok=True)

    pitcher_stats(cursor, pitchers, embeds)
    batter_stats(cursor, batters, embeds)


def pitcher_stats(cursor, players, embeds):
    pitcher_stats_raw = get_pitcher_stats(cursor, players)
    pitcher_stats_calc = calc_pitcher_stats(pitcher_stats_raw)
    plot_pitcher_stats(embeds, pitcher_stats_calc)


def batter_stats(cursor, players, embeds):
    batter_stats_raw = get_batter_stats(cursor, players)
    batter_stats_calc = calc_batter_stats(batter_stats_raw)
    plot_batter_stats(embeds, batter_stats_calc)


def plot_pitcher_stats(embeds, df):
    pitcher_keywords = {
        'ace': ['ERA', 'WHIP', 'QS%', 'GmSc', 'WAR-162'],
        'strikeouts': ['K-9', 'K-BB'],
        'walk': ['BB-9', 'K-BB'],
        'walks': ['BB-9', 'K-BB'],
        'dominant': ['ERA', 'WHIP', 'GmSc']
    }
    stat_to_idx = {'ERA': 4, 'WHIP': 5, 'K-9': 6, 'BB-9': 7, 'K-BB': 8, 'GmSc': 9, 'QS%': 10, 'WAR-162': 11}

    if df is None:
        df = pd.read_csv(f'{DATA_DIR}/stats/pitcher_stats_calc.csv', encoding='utf-8')
    player_data = {p.person_id: p for p in df.itertuples(index=False) if p.position == 'SP'}
    plot_stats(pitcher_keywords, stat_to_idx, embeds, player_data)


def plot_batter_stats(embeds, df):
    batter_keywords = {
        'slugger': ['SLG', 'ISO', 'HR%'],
        'contact': ['AVG'],
        'superstar': ['OPS', 'WAR-162'],
        'strikeout': ['K%'],
        'strikeouts': ['K%'],
        'clutch': ['Clutch']
    }
    stat_to_idx = {'AVG': 4, 'SLG': 6, 'ISO': 7, 'OPS': 8, 'K%': 9, 'HR%': 10, 'Clutch': 11, 'WAR-162': 12}

    if df is None:
        df = pd.read_csv(f'{DATA_DIR}/stats/batter_stats_calc.csv', encoding='utf-8')
    player_data = {p.person_id: p for p in df.itertuples(index=False)}

    plot_stats(batter_keywords, stat_to_idx, embeds, player_data)


def plot_stats(keywords, stat_index, embeds, player_data):
    players = [p for p in player_data]
    kw = get_keyword_similarity(keywords, players, embeds)
    labels = [p for p in player_data]
    for word, stats in keywords.items():
        cols = len(stats)
        fig, axs = plt.subplots(1, ncols=cols, figsize=(len(stats) * 6, 6), layout='tight')
        if cols == 1:
            axs.set_ylabel('Cosine similarity')
        else:
            axs[0].set_ylabel('Cosine similarity')
        y_vals = [float(kw[word][p]) for p in labels]
        all_points = []
        all_texts = []
        for i in range(cols):
            stat = stats[i]
            if cols == 1:
                ax = axs
            else:
                ax = axs[i]
            idx = stat_index[stat]
            ax.set_xlabel(stat)
            x_vals = [player_data[p][idx] for p in labels]
            all_points.append(ax.scatter(x_vals, y_vals, alpha = 0.7))
            m, b, r_value, p_value, std_err = linregress(x_vals, y_vals)
            ax.plot(np.unique(x_vals), m * np.unique(x_vals) + b, color='r')
            ax.set_title(f'$r^2$ = {r_value**2:.2f}, p = {p_value:.3f}')
            all_texts.append([ax.text(x_vals[j], y_vals[j], labels[j], ha='center', va='center', size='small') for j in
                     range(len(labels))])

        for i in range(cols):
            if cols == 1:
                ax = axs
            else:
                ax = axs[i]
            adjust_text(
                all_texts[i],
                objects=all_points[i],
                ax=ax,
                time_lim=3,
                arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                min_arrow_len=5,
            )

        plt.suptitle(
            f'Similarity of players to word "{word}" vs. {", ".join([x for x in stats])}',
            fontsize=14,
            fontweight='bold',
        )
        fig.savefig(f'figs/{word}_vs_{"_".join([x.lower() for x in stats])}.png')
        print(f'Plots for word {word} saved to {os.getcwd()}/figs/{word}_vs_{"_".join([x.lower() for x in stats])}.png.')

    plt.show()


def get_keyword_similarity(keywords, players, embeds):
    word_sims = {}
    for w in keywords:
        comps = {}
        for p in players:
            sim = embeds.similarity(w, p)
            comps[p] = sim
        word_sims[w] = {k: f'{v:.4f}' for k, v in
                              sorted(comps.items(), key=lambda item: item[1], reverse=True)}
    with open(f'similarities/keyword_sim.json', 'w', encoding='utf-8') as f:
        json.dump(word_sims, f, ensure_ascii=False, indent=4)

    return word_sims


def compare_war(stat_file, embeds):
    # load player stat file
    df = pd.read_csv(stat_file, encoding='utf-8')
    war_col = df.columns.tolist()[-1]
    df = df.rename(columns={war_col: 'war_162'})
    # for each unique position:
    for pos in df['position'].unique():
    # pick player with highest war/162 (for now)
        if pos == 'OF':
            pos_stats = df[df['position'].isin(['LF', 'CF', 'RF', 'OF'])]
        else:
            pos_stats = df[df['position'] == pos]
        key_player_idx = pos_stats['war_162'].idxmax()
    # get cosine similarity between player and all other players at position
        comps = []
        for item in pos_stats.itertuples():
            if item[0] == key_player_idx:
                key_player = item.person_id
                key_war = item.war_162
            else:
                comps.append((item.person_id, item.war_162))
        x_vals = [i[1] - key_war for i in comps]
        y_vals = [embeds.similarity(key_player, i[0]) for i in comps]
        labels = [i[0] for i in comps]
    # plot so y-axis is cosine sim, x-axis is (negative) war/162 diff
        fig, ax = plt.subplots(figsize=(12, 12), layout='tight')
        ax.set_xlabel('WAR/162 Diff.')
        ax.set_ylabel('Cosine similarity')
        ax.scatter(x_vals, y_vals, alpha=0.7)

        m, b, r_value, p_value, std_err = linregress(x_vals, y_vals)
        ax.plot(np.unique(x_vals), m * np.unique(x_vals) + b, color='r')
        ax.set_title(f'$r^2$ = {r_value ** 2:.2f}, p = {p_value:.3f}')

        texts = [plt.text(x_vals[i], y_vals[i], labels[i]) for i in range(len(labels))]
        adjust_text(texts)

        fig.suptitle(
            f'{pos} similarity vs. WAR to {key_player}',
            fontsize=14,
            fontweight='bold',
        )
        fig.savefig(f'figs/{pos}_vs_{key_player}.png')
        print(f'Plots for {pos} saved to {os.getcwd()}/figs/{pos}_vs_{key_player}.png.')

    plt.show()


# if __name__ == '__main__':
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()
#
#     random.seed(RANDOM_SEED)
#
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_colwidth', None)
#     pd.set_option('display.width', 200)
#
#     exp_num = '005'
#     exp_path = f'data/exp_{exp_num}' # not doing makedirs because it Has To exist by now
#
#     with ch_dir(exp_path):
#         for model_path in glob.glob('*.model'):
#             model = Word2Vec.load(model_path)
#             # get_stats(cursor, model.wv)
#             for path in glob.glob('stats/*calc.csv'):
#                 compare_war(path, model.wv)