# imports
import os
import json
import random
from tqdm import tqdm
from gensim.models import Word2Vec


# data paths
TKN_TEXT_STORAGE = 'data/texts/tokenized'
MENT_TEXT_STORAGE = 'data/texts/mention_replaced'
RANDOM_SEED = 42


class DataCorpus:
    def __init__(self, dirname, out_path, player_ids):
        self.dirname = dirname
        self.loc = out_path
        self.players = player_ids

    def build(self):
        docs = []
        for fname in tqdm(os.listdir(self.dirname)):
            with open(os.path.join(self.dirname, fname), 'r', encoding='utf-8') as target:
                docs.append(target.read())
        with open(self.loc, 'w', encoding='utf-8') as f:
            f.write('\n'.join(docs))

    def __iter__(self):
        with open(self.loc, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()


# corpus building
def build_corpus(exp_num, corpus_type='token'):
    if corpus_type == 'mention':
        dirname = MENT_TEXT_STORAGE
    else:
        dirname = TKN_TEXT_STORAGE
    exp_path = f'data/exp_{exp_num}'
    with open(f'{exp_path}/player_ments_{exp_num}.json', 'r', encoding='utf-8') as f:
        player_dict = json.load(f)
    player_ids = list(player_dict)
    corpus_path = f'{exp_path}/{corpus_type}_corpus_{exp_num}.cor'
    corpus = DataCorpus(dirname, corpus_path, player_ids)
    corpus.build()
    return corpus


# def load_corpus(exp_num, corpus_type='token'):
#     corpus_path = f'data/exp_{exp_num}/{corpus_type}_corpus_{exp_num}.cor'
#     corpus = DataCorpus(corpus_path)
#     return corpus


# actual embeddings
def train_word2vec(exp_num, corpus, corpus_type):
    out_path = f'data/exp_{exp_num}/w2v_{corpus_type}_{exp_num}.model'
    model = Word2Vec(
        sentences=corpus,
        seed=RANDOM_SEED,
        workers=1,
        sg=1,
        vector_size=100,
        alpha=0.025,
        window=5,
        negative=5,
        epochs=1,
        sample=1e-05
    )
    print('Word2Vec embeddings trained.')
    model.save(out_path)
    print(f'Embedding model saved at {out_path}.')
    return model


def load_word2vec(exp_num, corpus_type):
    model_path = f'data/exp_{exp_num}/w2v_{corpus_type}_{exp_num}.model'
    model = Word2Vec.load(model_path)
    return model


if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    # mention_corpus = build_corpus('001', corpus_type='mention')
    with open(f'data/exp_001/player_ments_001.json', 'r', encoding='utf-8') as f:
        player_dict = json.load(f)
    player_ids = list(player_dict)
    mention_corpus = DataCorpus(MENT_TEXT_STORAGE, 'data/exp_001/mention_corpus_001.cor', player_ids)
    # mention_corpus = load_corpus('001', 'mention')
    # model = train_word2vec('001', mention_corpus, 'mention')
    model = load_word2vec('001', 'mention')
    embeds = model.wv
    print(f'Vocab size: {len(embeds.index_to_key)}')
    players = mention_corpus.players
    similar_words = {}
    similar_players = {}
    for p in players:
        word_results = embeds.most_similar(positive=[p], topn=10)
        similar_words[p] = {word_key: f'{word_sim:.4f}' for word_key, word_sim in word_results}
        sim_list = [i for i in players if i != p]
        comps = {}
        for comp in sim_list:
            sim = embeds.similarity(p, comp)
            comps[comp] = sim
        similar_players[p] = {k: f'{v:.4f}' for k, v in sorted(comps.items(), key=lambda item: item[1], reverse=True)[:10]}
    with open('data/exp_001/similar_words_cosine.json', 'w', encoding='utf-8') as f:
        json.dump(similar_words, f, ensure_ascii=False, indent=4)
    with open('data/exp_001/similar_players_cosine.json', 'w', encoding='utf-8') as f:
        json.dump(similar_players, f, ensure_ascii=False, indent=4)