# Using the WAR Visualizer

The information below can also be found in the `README.md` file of the [projector repository](https://github.com/ledibr/words-above-replacement-visualizer).

### Tensors

- **WAR Word2Vec 10K:** Embeddings of 10k most frequent words in the model vocabulary.
- **WAR Word2Vec 12K:** Embeddings of 12k most frequent words in the model vocabulary.
- **WAR Word2Vec Players + Target Words:** Embeddings of studied player set (141 players)
  as well as all "interest words" (256 words).

### Metadata

Each embedding is associated with several metadata fields that can be used as labels
or color categories in the projector.
- **word:** The token corresponding to the embedding.
- **type:** Whether the token is an entity (player) or a non-entity word.
- **count:** The count of the token in the corpus.

  The following two fields are exclusive to entities; non-entities have a value of 'N/A'.
  (Coloring by these fields is most useful when using the "WAR Word2Vec Players + Target Words" tensors 
  or isolating all entity embeddings using 'Search by type' → 'player'.)

- **position:** The primary position of the player.
- **team:** The primary team of the player.