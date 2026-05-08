# Words Above Replacement: A Computational Approach to Analyzing Baseball Writing

This repository contains all code and files for the submission of my capstone project,
*Words Above Replacement: A Computational Approach to Analyzing Baseball Writing*, as part of
the completion of the CLMS program at Brandeis University.

All code was run in PyCharm on a Framework Laptop 13 with the following specifications:
- Python version 3.12.4
- OS: Windows 11 Home version 25H2
- CPU: AMD Ryzen 5 7640U (3.50 GHz)
- GPU: AMD Radeon 760M (integrated)
- RAM: 16 GB

## Repository

Project requirements are listed in `requirements.txt`.

The project's paper, which describes the project's background, methodology, and results,
is contained in the [words-above-replacement-paper.md](words-above-replacement-paper.md) file.

All files generated during training and analysis of the main experiment are located in 
the [`results` directory](results).

An interactive visualization of the embeddings in the repository using the TensorBoard Projector
framework can be found at https://ledibr.github.io/words-above-replacement-visualizer/. The repository
for this GitHub Pages site can be found at https://github.com/ledibr/words-above-replacement-visualizer.

For more information on the project and specific details on the repository's contents,
see the [index](docs/index.md) in the [docs folder](docs).

## Running the demo

The `demo.py` file contains a script that replicates the embedding training and results visualization
pipeline, beginning with a pre-existing `mention_corpus.cor` corpus file.

To run the demo, first clone this repository to your local machine (or fork it and then
clone that, or really whatever you want to do to achieve the same end goal). Once available,
run the following command while in the root directory:
```
python -m demo
```
or alternately:
```
python demo.py
```
The demo produces a number of files in a new `demo_files` directory, which generally correspond 
to the experiment files in the `results` directory. For a description of these files, see the
"Results" section in [`index.md`](docs/index.md).

The primary exception to this rule is the `vis` sub-directory, which contains files for use with 
the public TensorBoard Projector located at https://projector.tensorflow.org/. To use, select the 
"Load" button and upload the tensor file and (optionally) one of the two metadata files, 
per the site's directions. The files include:
- `war_word2vec_demo.model`: Saved `KeyedVectors` of selected vocab (player set + interest words).
- `war_word2vec_demo_metadata.tsv`: Basic tensor metadata containing only vocab labels.
- `war_word2vec_demo_metadata_annotated.tsv`: Expanded tensor metadata containing "word", "type", "count", "position", and "team" fields.
- `war_word2vec_demo_tensor.tsv`: Tensor file of selected vocab.
- `word_counts_demo.csv`: Counts of tensor vocab.

While the `vis` files do not have equivalents in the `results` directory, they are equivalent to the
`war_word2vec_focused` files/"WAR Word2Vec Players + Target Words" data in the project visualizer 
mentioned in the [Repository](#repository) section above. This allows you to replicate the embedding
projections (for the target embeddings) without relying on my dedicated demo repository.