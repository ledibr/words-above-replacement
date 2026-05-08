import os, json, random, logging, sys, glob, bz2

from gensim.models.keyedvectors import load_word2vec_format
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import gensim.scripts.word2vec2tensor
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from utils import DATA_DIR, DATABASE, ARTICLE_STORAGE, PARSED_TEXT_STORAGE, CLEAN_TEXT_STORAGE, TKN_TEXT_STORAGE, MASK_TEXT_STORAGE, MENT_TEXT_STORAGE, RANDOM_SEED, ch_dir


class LossCallback(CallbackAny2Vec):
    def __init__(self, path):
        self.epoch = 1
        self.prev_loss = 0
        self.path = path

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss() * 1e-7
        curr_loss = loss - self.prev_loss
        self.prev_loss = curr_loss
        # print(f'Loss after epoch {self.epoch}: {curr_loss:.5f}')
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'Loss after epoch {self.epoch}: {curr_loss:.5f}\n')
        self.epoch += 1


class DataCorpus:
    def __init__(self, dirname, out_path, player_ids):
        self.dirname = dirname
        self.loc = out_path
        self.players = player_ids

        if not os.path.exists(out_path):
            docs = []
            for fname in tqdm(os.listdir(self.dirname)):
                with open(os.path.join(self.dirname, fname), 'r', encoding='utf-8') as target:
                    docs.append(target.read())
            with open(self.loc, 'w', encoding='utf-8') as f:
                f.write('\n'.join(docs))

    def __iter__(self):
        if Path(self.loc).suffix == '.bz2':
            with bz2.open(self.loc, 'rt', encoding='utf-8') as f:
                for line in f:
                    yield line.strip().split()
        else:
            with open(self.loc, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip().split()


def train_word2vec(corpus):
    dim = 300
    lr = 0.025
    window = 10
    neg = 5
    ep = 3
    sample = 1e-5

    params = f'dim-{dim}_lr-{lr}_window-{window}_neg-{neg}_ep-{ep}_sample-{sample}'

    logging.basicConfig(
        filename=f'LOGGER_{params}.log',
        filemode='w',
        encoding='utf-8',
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        force=True
    )
    cs_handler = logging.StreamHandler(sys.stdout)
    cs_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    logging.getLogger().addHandler(cs_handler)

    model = Word2Vec(
        sentences=corpus,
        seed=RANDOM_SEED,
        workers=1,
        sg=1,
        min_count=10,
        vector_size=dim,
        alpha=lr,
        window=window,
        negative=neg,
        epochs=ep,
        sample=sample,
        compute_loss=True,
        callbacks=[LossCallback(f'LOSS_{params}.log')]
    )
    print(f'Word2Vec embeddings trained with config: dim = {dim}, lr = {lr}, window = {window}, neg = {neg}, ep = {ep}, sample = {sample}')

    # Model saving disabled for demo purposes
    # out_path = f'w2v_{params}.model'
    # model.save(out_path)
    # print(f'Embedding model saved at {os.getcwd()}/{out_path}.')

    print(f'Vocab size: {len(model.wv.index_to_key)}')
    print(f'Player count: {len(corpus.players)}')

    os.makedirs('similarities', exist_ok=True)
    get_similar_words(model.wv, corpus.players)
    get_similar_players(model.wv, corpus.players)

    return model


def grid_search(exp_num, corpus):
    vec_sizes = [300]
    alphas = [0.025]
    window_sizes = [10]
    neg_vals = [5]
    ep_vals = [7, 10]
    sample_vals = [1e-5]

    out_dir = f'data/exp_{exp_num}/grid_search'
    os.makedirs(f'{out_dir}', exist_ok=True)
    with open(f'{out_dir}/gs_data.txt', 'a', encoding='utf-8') as f:
        f.write(f'Grid search values:\nVector sizes: {vec_sizes}\nAlphas: {alphas}\nWindow sizes: {window_sizes}\nNeg values: {neg_vals}\nEpochs: {ep_vals}\nSample values: {sample_vals}\n')

    for dim in vec_sizes:
        for lr in alphas:
            for window in window_sizes:
                for neg in neg_vals:
                    for ep in ep_vals:
                        for sample in sample_vals:
                            params = f'dim-{dim}_lr-{lr}_window-{window}_neg-{neg}_ep-{ep}_sample-{sample}'
                            os.makedirs(f'{out_dir}/{params}', exist_ok=True)

                            logging.basicConfig(
                                filename=f'{out_dir}/{params}/LOGGER_{exp_num}.log',
                                filemode='w',
                                encoding='utf-8',
                                format='%(asctime)s : %(levelname)s : %(message)s',
                                level=logging.INFO,
                                force=True
                            )
                            cs_handler = logging.StreamHandler(sys.stdout)
                            cs_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
                            logging.getLogger().addHandler(cs_handler)

                            model = Word2Vec(
                                # sentences=corpus,
                                corpus_file=corpus.loc,
                                seed=RANDOM_SEED,
                                workers=1,
                                sg=1,
                                min_count=10,
                                vector_size=dim,
                                alpha=lr,
                                window=window,
                                negative=neg,
                                epochs=ep,
                                sample=sample,
                                compute_loss=True,
                                callbacks=[LossCallback(f'{out_dir}/{params}/LOSS_{exp_num}.log')]
                            )
                            print(
                                f'Word2Vec embeddings trained with config: dim = {dim}, lr = {lr}, window = {window}, neg = {neg}, ep = {ep}, sample = {sample}')

                            out_path = f'{out_dir}/{params}/w2v_{exp_num}_gs.model'
                            model.save(out_path)
                            print(f'Embedding model saved at {out_path}.')
                            # get_similar_words(f'{out_dir}/{params}', model.wv, corpus.players)
                            get_similar_players(f'{out_dir}/{params}', model.wv, corpus.players)


def get_similar_words(embeds, players):
    with open(f'{DATA_DIR}/target_words.json', 'r', encoding='utf-8') as f:
        words = json.load(f)
    word_list = [w for label in words for w in words[label]]
    cos_words = {}

    for p in players:
        comps = {}
        for w in word_list:
            if embeds.has_index_for(w):
                sim = embeds.similarity(p, w)
                comps[w] = sim
            else:
                print(f'No vector found for word {w}.')
        cos_words[p] = {k: f'{v:.4f}' for k, v in
                              sorted(comps.items(), key=lambda item: item[1], reverse=True)[:10]}
    with open(f'similarities/similar_words.json', 'w', encoding='utf-8') as f:
        json.dump(cos_words, f, ensure_ascii=False, indent=4)
    print(f'Top 10 most similar words per player written to {os.getcwd()}/similarities/similar_words.json.')

    categorize_similar_words(words, cos_words)

    for label in ['positive', 'negative', 'stats', 'gameplay', 'physical']:
        cos_words = {}
        for p in players:
            comps = {}
            for w in words[label]:
                sim = embeds.similarity(p, w)
                comps[w] = sim
            cos_words[p] = {k: f'{v:.4f}' for k, v in
                            sorted(comps.items(), key=lambda item: item[1], reverse=True)[:10]}
        with open(f'similarities/similar_words_{label}.json', 'w', encoding='utf-8') as f:
            json.dump(cos_words, f, ensure_ascii=False, indent=4)
        print(f'Top 10 most similar {label} words per player written to {os.getcwd()}/similarities/similar_words_{label}.json.')

    # return cos_words


def get_similar_players(embeds, players):
    similar_players = {}
    for p in players:
        sim_list = [i for i in players if i != p]
        comps = {}
        for comp in sim_list:
            sim = embeds.similarity(p, comp)
            comps[comp] = sim
        similar_players[p] = {k: f'{v:.4f}' for k, v in sorted(comps.items(), key=lambda item: item[1], reverse=True)[:10]}
    with open(f'similarities/similar_players.json', 'w', encoding='utf-8') as f:
        json.dump(similar_players, f, ensure_ascii=False, indent=4)
    print(f'Top 10 most similar players per player written to {os.getcwd()}/similarities/similar_players.json.')

    # return similar_players


def categorize_similar_words(words, player_words):
    word2cat = {}
    for k, v in words.items():
        for word in v:
            word2cat[word] = k

    pcats = {p: Counter() for p in player_words}
    for p, l in player_words.items():
        for w in l:
            pcats[p][word2cat[w]] += 1

    data = {'person_id': [p for p in pcats]}
    for k in words:
        data[k] = [c[k] for c in pcats.values()]
    df = pd.DataFrame(data)
    df.to_csv('similarities/player_word_stats.tsv', sep='\t', index=False, mode='w', encoding='utf-8')
    print(f'Category stats for player-word similarities written to {os.getcwd()}/similarities/player_word_stats.tsv.')


if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    exp_num = '005'
    exp_path = f'data/exp_{exp_num}'

    # with open('data/mention_corpus.cor', 'rb') as f:
    #     with bz2.open('data/mention_corpus.cor.bz2', 'wb') as bz:
    #         bz.write(f.read())

    # with ch_dir(exp_path):
    #     with open(f'player_list.txt', 'r', encoding='utf-8') as f:
    #         player_ids = [line.strip() for line in f]
    #     mention_corpus = DataCorpus(f'{MENT_TEXT_STORAGE}/special', f'mention_corpus.cor', player_ids)
        # model = train_word2vec(mention_corpus)
        # for model_path in glob.glob('*.model'):
        #     model = Word2Vec.load(model_path)
            # get_similar_words(model.wv, player_ids)
            # get_similar_players(model.wv, player_ids)

    # grid_search(exp_num, mention_corpus)