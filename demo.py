

# load corpus from corpus file
# train the embeddings w selected hyperparameters
# generate all the files and images and shit
# save tensors/metadata for players + words that can be loaded into the embedding projector


# model = Word2Vec.load('data/exp_005/w2v_dim-300_lr-0.025_window-10_neg-5_ep-3_sample-1e-05.model')

# with open('data/exp_005/player_list.txt', 'r', encoding='utf-8') as f:
#     players = [line.strip() for line in f]
# with open('data/target_words.json', 'r', encoding='utf-8') as f:
#     words = json.load(f)
# words = [w for label in words for w in words[label]]
# targets = players + words
# vectors = np.array([model.wv.get_vector(t) for t in targets])

# counts = {'word': [], 'count': []}
# for index, word in enumerate(targets):
#     counts['word'].append(word)
#     counts['count'].append(model.wv.get_vecattr(word, 'count'))
# df = pd.DataFrame.from_dict(counts)
# df.to_csv(f'{exp_path}/word_counts_focused.csv', index=False, mode='w', encoding='utf-8')

# model.wv.save_word2vec_format(fname=f'{exp_path}/war_word2vec_del.model')
# outvecs = KeyedVectors.load_word2vec_format(f'{exp_path}/war_word2vec_del.model', limit=10001)
# kv = KeyedVectors(vector_size=model.wv.vector_size)
# kv.add_vectors(targets, [model.wv.get_vector(t) for t in targets])
# kv.save_word2vec_format(fname=f'data/vis_models/war_word2vec_focused.model')