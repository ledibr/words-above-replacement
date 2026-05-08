#!/usr/bin/env python3

import os, random, glob, json
import pandas as pd
from gensim.models.word2vec import KeyedVectors
from gensim.scripts.word2vec2tensor import word2vec2tensor
from embeddings import DataCorpus, train_word2vec
from clustering import cluster_embeddings
from stat_plots import plot_pitcher_stats, plot_batter_stats, compare_war
from utils import DATA_DIR, RANDOM_SEED, ch_dir


def get_word_counts(targets, embeds):
    counts = {'word': [], 'count': []}
    for index, word in enumerate(targets):
        counts['word'].append(word)
        counts['count'].append(embeds.get_vecattr(word, 'count'))
    df = pd.DataFrame.from_dict(counts)
    df.to_csv(f'vis/word_counts_demo.csv', index=False, mode='w', encoding='utf-8')
    print(f'Tensor vocab counts written to {os.getcwd()}/vis/word_counts_demo.csv.')


def annotate_metadata():
    with open(f'{DATA_DIR}/player_list.txt', 'r', encoding='utf-8') as f:
        players = [line.strip() for line in f]

    metadata_path = f'vis/war_word2vec_demo_metadata'
    metadata_df = pd.read_csv(f'{metadata_path}.tsv', encoding='utf-8', sep='\t', names=['word'])
    wordcount_df = pd.read_csv(f'vis/word_counts_demo.csv', encoding='utf-8')
    batter_df = pd.read_csv(f'{DATA_DIR}/stats/batter_stats_calc.csv', encoding='utf-8', header=0, names=['word', 'position', 'team'], usecols=[0, 1, 3])
    pitcher_df = pd.read_csv(f'{DATA_DIR}/stats/pitcher_stats_calc.csv', encoding='utf-8', header=0, names=['word', 'position', 'team'], usecols=[0, 1, 3])
    player_df = pd.concat([batter_df, pitcher_df])

    metadata_df['type'] = ['player' if w in players else 'word' for w in metadata_df['word']]
    metadata_df = pd.merge(metadata_df, wordcount_df, how='left', on='word')
    metadata_df = pd.merge(metadata_df, player_df, how='left', on='word').fillna('N/A')

    metadata_df.to_csv(f'{metadata_path}_annotated.tsv', sep='\t', index=False, mode='w', encoding='utf-8')
    print(f'Annotated tensor metadata written to {os.getcwd()}/{metadata_path}_annotated.tsv.')


if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    os.makedirs('demo_files', exist_ok=True)
    with open(f'{DATA_DIR}/player_list.txt', 'r', encoding='utf-8') as f:
        player_ids = [line.strip() for line in f]
    # NOTE: the dirname here won't be used, so it doesn't matter
    mention_corpus = DataCorpus('DEMO', f'{DATA_DIR}/mention_corpus.cor.bz2', player_ids)
    with ch_dir('demo_files'):
        model = train_word2vec(mention_corpus)
        embeds = model.wv
        os.makedirs('clusters', exist_ok=True)
        cluster_embeddings(embeds)

        os.makedirs('figs', exist_ok=True)
        plot_pitcher_stats(embeds, df=None)
        plot_batter_stats(embeds, df=None)
        for path in glob.glob(f'{DATA_DIR}/stats/*calc.csv'):
            compare_war(path, embeds)

        os.makedirs('vis', exist_ok=True)
        with open(f'{DATA_DIR}/target_words.json', 'r', encoding='utf-8') as f:
            words = json.load(f)
        words = [w for label in words for w in words[label]]
        targets = player_ids + words
        kv = KeyedVectors(vector_size=embeds.vector_size)
        kv.add_vectors(targets, [embeds.get_vector(t) for t in targets])
        kv.save_word2vec_format(fname='vis/war_word2vec_demo.model')
        print(f'Target word vectors written to {os.getcwd()}/vis/war_word2vec_demo.model.tsv.')
        word2vec2tensor('vis/war_word2vec_demo.model', 'vis/war_word2vec_demo')
        print(f'Tensors and metadata for visualization written to {os.getcwd()}/vis/ directory.')
        get_word_counts(targets, embeds)
        annotate_metadata()