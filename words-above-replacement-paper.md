# Words Above Replacement: A Computational Approach to Analyzing Baseball Writing
**Author:** Lindsay Dial (ledibr)  
**Affiliation:** Michtom School of Computer Science, Brandeis University  
_Capstone project submitted in partial fulfillment of the requirements for the degree of Master of Science in Computational Linguistics._

## Introduction

- motivation/goals
  - building distributed semantic representations of baseball entities (players) 
        to analyze attitudes and trends in media coverage
  - From the dawn of professional baseball, the way a player appears in the papers 
    has mattered to the public as much as their performance on the field. 
    Sportswriting has always involved judgments of statistics as well as character, 
    and evidence of bias—especially racial bias—has been well-documented 
    in everything from blog posts to internal scouting reports. 
    But there is only so much that we, as human readers, can discern, 
    which made me wonder: what if there is more to the media’s language 
    than meets the eye? What sentiments are expressed that we may not easily perceive? 
    And is it possible for us to learn something new about a player’s potential 
    from the way that people talk about them?
  - Expanding the very small field of baseball NLP to advance sabermetrics
    and support future research
- research questions
  - What players are written about most similarly?
  - Do these similarities appear based on race, nationality, position, team, or other factors?
  - What particular words, whether "objective" descriptors or more "subjective"
    personal commentary, are most closely associated with certain players (and groups)?
  - How does language correlate with play-based statistics, and can past writing
    predict future performance?
- Using publicly available writing from the last ten years...
- I then use them to explore a number of questions, including... 
- By using broader linguistic context, my research will provide unprecedented insight
  into how MLB players are depicted in the kind of writing that millions of fans 
  read every day. To my knowledge, this kind of large-scale computational analysis 
  of both bias and predictive potential in public-facing baseball writing 
  has never been performed before.
- The contributions of this paper are as follows: to ... , I ...
  In addition to evaluating ... , I also explore ... 
  Finally, I ... by ...
  My results ... and I propose future directions for expanding my findings. 
- My discussion begins with a brief overview of prior work in this area in
  ... I then describe my data and outline ... and ... , followed by 
  an analysis of my results in ... Finally, I review my conclusions, contributions, 
  and plans for future work in ... , as well as addressing the project's limitations ...


## Related Work
- focus on yamada et al., rob arthur, anything tagged as especially
    useful in the zotero

### Entity Embeddings
- primarily wikipedia2vec, can mention other research

### Bias in Sports
- issues of note: stacking, prospect promotion, media bias

### Baseball NLP
- While NLP techniques have been used to evaluate bias in other sports media, 
  and baseball journalism has been examined through a non-computational lens, 
  little work has been performed on baseball media at significant scale. 
  Rob Arthur and Ben Lindbergh’s 2019 research on Cincinnati Reds scouting reports 
  from 1991–2003 looked at word choice in addition to numerical grades and 
  utilized sentiment analysis techniques, but did not involve full-text NLP.
- Existing work on baseball NLP has focused mainly on prospect promotion,
  hasn't been very advanced in terms of actual prediction systems or anything
- trouble with the curve
- reds scouting reports
- prospect hunting with deep learning


## Data

The data for this project was sourced from a SQL dump of tables from Baseball Reference (Sports Reference LLC, 2025)
provided by Sean Forman. The SQL dump contains tables of player biographical information
and statistics by season, team information and statistics by season, and information on
articles linked to player pages from their internal feed. Baseball Reference (hereafter referred to as BRef) 
maintains a directory of sites in a SQL table that contribute to the player newsfeed; articles are linked
to players based on site authors' usage of the BRef "Player Linker Functionality", and the article-player
associations are stored in another SQL table. A third SQL table holds information on each individual article,
including the associated site ID, article URL, and date inserted. The database provided to me contains data
up to December 29, 2025.

To make the data usable, I converted the dump file from its MariaDB-compatible format to a SQLite-parsable dialect
and imported it into a database. I then created two new tables compiling the information necessary
to begin retrieving article data and identifying players:
- **new_articles_players:** A table mapping article IDs to player IDs, correcting instances of multiple different IDs
  for the same player appearing and removing duplicates.
- **new_player_data:** A table listing each player's ID, total number of articles linked to them, and the years of their
  first and last seasons (through 2025).

The text corpus used in the project's central experiments consists of 121,816 unannotated articles 
from 49 different sites published between 2016 and 2025. To collect article data, I downloaded webpages from the 
article URLs provided in the BRef database using the `requests` library. The articles were first filtered based on 
the following criteria:
- Article inserted into database no earlier than January 1, 2016
- Parent website's latest entry in the database no earlier than December 2025 (to reduce the risk of trying to pull
  from dead sites and prioritize actively-read websites)
- Article ID associated with player ID of a player with at least 1,000 articles (to focus on players with a significant
  amount of associated data)

These criteria produced a set of 218,758 articles. Before fixing any request errors, 145,770 of these webpages were
successfully requested. A SQL table called 'request_data' was created to keep track of relevant metadata: article ID,
article URL, timestamp of request/modification, download status (as a boolean success indicator, later an integer value
indicating stage of text processing), name of storage file, site HTTP headers, and request exceptions (if any arose).
A number of issues arose during the article collection process, including: `TooManyRequests`, `ConnectionError`, 
`Timeout`, `HTTPError`, and other exceptions raised by the `requests` library; webpages with captchas that
allegedly downloaded successfully, but in reality only displayed text indicating the captcha's presence; invalid URLs;
paywalls; and sites that did not throw a logged exception, but did not return any content when downloaded.
When possible, I attempted to resolve these errors and re-download articles, successfully increasing the number of
downloaded articles to 183,617; when the error could not be resolved, the article was removed from the target set.
Some successfully downloaded webpages were also removed based on their content, including those with irrelevant content,
extremely difficult-to-parse HTML, and pages from non-English websites.

After downloading all webpage content and correcting errors, I examined the HTML of each distinct website to identify
the tags used for the main content. Using that information, I parsed all downloaded articles to extract their body 
content, excluding elements like the title and author(s). To ensure the parsing operated correctly, I parsed a small
sample of articles from each website and tweaked the process as necessary upon encountering errors until I felt
confident in running the parsing script on the entire set of articles from that site.

After extracting the article content from its HTML, I removed more articles from the target set, including: articles
with files under 400 bytes, catching empty articles and extremely short blog posts; articles from sites with fewer than
100 articles; and articles from sites with generally low-quality writing (e.g. frequent spelling errors). I then
cleaned up the article content by systematically removing pieces of extraneous text via simple text filtering, using
the `re` library for regex functionality. Common "problem strings", such as image captions, "Thanks for reading!"
messages, and un-embedded links, were identified for each site during the HTML parsing process. Final preprocessing
of the text data was achieved by using `gensim` functions to perform standard tokenization, lowercasing,
Unicode normalization and de-accenting, and stripping extraneous whitespace. Stopwords were not removed from the data.

The final corpus used in the main experiments consists of 121,816 documents containing 72,043,079 tokens, 83,821
unique types, and 714,086 identified mentions of players in the target set. One additional team-specific site was
used in some early experiments and the grid search process, but was ultimately excluded from the data to reduce
dataset imbalance, as it had 22,063 articles present in the corpus (for a total corpus size of 143,879 documents).
A full list of the sites in the corpus, their number of articles, and their team focus (if any) can be found in
the [Article Statistics doc](docs/articles.md).

## Embeddings
- methodology for text replacement
  - replacing player mentions w/ unique token allows unification of all mentions
    in one form as well as comparison to other single-token words/mapping into
    same space
  - To identify sets of players for experiments: re-filter the players to those 
    with a minimum number of articles based on the IDs of the parsed articles, 
    storing new counts in the updated_player_data SQL table. 
  - Some players are excluded from experiments due to “problem names”, 
    e.g. names with internal punctuation, more than two tokens, or identical full names. 
  - Team info masked to limit their influence on player embeddings 
    for the purposes of clustering. Location, noun (singular and plural, plus 
    alternate forms as necessary), abbreviation, and leagues are all masked with 
    placeholders (e.g. ##TEAM) before player mentions are replaced. 
  - Preparing player mentions: for each player mentioned in a given article—
    identified before beginning the process due to article-player mapping in the 
    new_articles_players SQL table—replace each mention of them (by full name or 
    only last name; in select cases, nicknames may be included in the list) in 
    the article text with their unique person ID from the database. 
  - Like removing the extraneous “problem strings”, mention replacement can be 
    performed with relatively simple regex. 
  - To account for missing player tags, all players in the set of interest 
    are searched for by full name in the article and added to the list of 
    mentioned players if found before proceeding as usual. 
  - Articles with no mentions are left in the corpus to provide additional data 
    for non-mention word embeddings.

- grid search process (incl. smaller player set used for testing)
  - mention papers relevant to selecting hyperparams
  - Second experiment/grid search: smaller, manually-selected set of 24 players, 
      all heavily represented in the data, distributed across a select set of teams 
      to get approximately equal representation of factors like team, position, etc. 
    - Still avoiding punctuated problem names, but not worrying much about duplicate names 
    - Picked from approx. 1 team per division, so primarily getting groups of ~4 
      players per team/division + per position (pitcher, shortstop, outfielder, 
      other non-catcher position player) as well as a mix of race/nationality/years/etc.
  - Third experiment: further grid search, same player set 
- training process (incl. params)
  - Embeddings are implemented by using the Python gensim library to train a 
      Word2Vec skip-gram model on the textual data. 
  - Training is single-threaded for reproducibility (along with the use of a 
    set random seed). 
  - Hyperparameters for primary experiments: 
    - 300d
    - initial learning rate = 0.025
    - window = 10
    - negative sample value = 5
    - downsampling threshold: 1e-5
    - 3 epochs 
  - Loss tracked during the training process as a measure of convergence 
    for epoch values

## Experiments
- describe specific experimental setup for main stuff here
  - First experiment: manually-selected set of 115 players with at least 700 articles, 
    distributed across MLB teams (may not even need to mention)
  - Fourth experiment: same 24-player set with chosen hyperparameters, but without 
    Metsmerized data 
  - Fifth experiment: set of 141 players with at least 1000 articles 
    (excluding Metsmerized), only curated to remove “problem names”

- e.g. embedding info, corpus subset, things tested/plotted, etc.
  - Using the embeddings for analysis includes computing similarity metrics 
    between players, as well as between players and non-player words; 
    clustering player vectors; and examining correlations between player vectors 
    and statistical metrics like WAR.
  - Similarity metrics are primarily calculated using cosine similarity. 
  - Comparing players to all words was generally uninformative, so a narrowed set 
    of interest words was selected by examining high-frequency vocabulary in the 
    full corpus, as well as words identified as prominently associated with 
    different classes of players (e.g. race, position) in the literature. 
  - Words classified into eight general categories: positive, negative, stats, 
    gameplay, physical, status, contract, nationality 
  - Similarity calculations for each player (always top 10 most similar):
    - Other players 
    - All words from interest set 
    - All words for each category in: positive, negative, stats, gameplay, physical 
  - Clustering:
    - Purpose: exploring "natural" groupings of players in the data
    - K-means and agglomerative clustering (average linkage w/ cosine distance)
      performed using sklearn 
    - Number of clusters tuned on k-means (value used for both), linkage/distance 
      metrics tuned for agglomerative; all tuning evaluated using silhouette score 
      (scored w/ cosine distance for # clusters, model’s distance metric for 
      linkage/distance metrics)
    - Cluster silhouette scores and scatter plot of players for each method 
      created with matplotlib 
    - Player embeddings reduced to 2 dimensions for plotting with IncrementalPCA 
      from sklearn 
  - Stat analysis:
    - Selected common words from player-word similarities and identified 
      player stats that might be associated with them 
    - For each word, plotted similarity between all players (split into position 
      players vs starting pitchers, relievers excluded) and word vs. each relevant 
      statistic (e.g. ‘slugger’ vs. SLG, ISO, HR%) using matplotlib 
    - r^2 and p-value reported for line of best fit using scipy

<!-- TO DO: Examine WAR correlations by selecting a set of player pairs/one player 
and a set of comparison players, plotting entity embedding similarity/distance vs. 
WAR/162 (or other metric) differential, and examining fit -->

- also include charts of e.g. player composition

## Results
- clustering
  - clusters really like grouping by team as well as position
  - [post graphs here]
  - agglomerative generally more coherent than k-means (better avg silhouette score)
- similarity
- various things i compared
- alllllllllll the images
- Visualization: Tensorboard Projector on GitHub Pages site (separate repo)


## Conclusion
- well, conclusion
- future work (probably will be most of it)
  - Expanding dataset:
    - Getting articles from FanGraphs (dump from Dan Szymborski), MLB, etc.
      - MLB attaches articles to player IDs - “news” in URL (vs. “video”, etc.)
    - Incorporating prospect scouting reports (TWTC has a corpus, though its formatting has errors)
    - Creating a "test set" using annotated documents to try predicting entity mentions using the embeddings
  - Improving entities/getting higher quality embeddings that capture more info:
    - NER to identify player names, since the regex is imperfect
    - Coreference resolution (to help get pronouns, maybe nicknames?)
    - Entity disambiguation (will help with siblings, parents, ppl who randomly have same name)
    - Knowledge graph linking (as in Wikipedia2Vec)
  - Predictive analysis:
    - Examining trends over time related to WAR/other stats (i.e. embeddings from one time period compared to stats
      of next year)
    - Combining embeddings w/ existing player projection system (e.g. ZiPS, since there's a good chance I will be
      able to get access to this and work w/ Dan Szymborski) to see if it can improve predictions
      - explain ZiPS


## Limitations
- Dataset:
  - Limited in scope to those sites aggregated by Baseball Reference, 
  which are mostly less popular and lower in quality than “mainstream” sources 
  - Heavily skewed towards certain teams (and by extension players)
  - Imperfections in spelling, player identification (by BRef), parsing, etc. 
- Mention identification challenges:
  - Players not being tagged correctly by BRef (both false positives and false negatives)
  - Spelling issues 
    - Happen all the time everywhere, even in mainstream/professional publications 
    - Hopefully not prevalent enough to cause meaningful issues 
  - Nicknames 
    - Some players frequently referred to by nicknames (shortened version of name 
      or otherwise), e.g. Belli, Vladdy, Alvy, The Rock/La Piedra, Ern Dog 
    - Nickname not always in BRef database, can’t be automated 
    - May overlap with “normal” words (more of a problem w/ historical players 
      not in the player set anyways), e.g. “Lefty”, “Junior” 
    - Would have to handle via manual curation for list matching 
  - Shared names 
    - General coincidences/common names 
      - Players w/ same last name (e.g. Chapman, Marte, Rosario, Smith, Turner, Alvarez)
      - Players w/ same full name (e.g. Max Muncy, Jose Ramirez, Luis Castillo)
      - Players w/ same first name (e.g. Miguel, Jose, Matt, Trevor)
      - Other special cases (e.g. Schneider)
    - Family (usu. siblings or fathers who also played)
      - Any “Jr.” (e.g. Guerrero, Tatis)
      - Active siblings (e.g. Acuña, Contreras)
      - Famous family w/o Jr. (e.g. Bichette)
    - Other “problem names” 
      - Some names have special characters that will likely be split in tokenization 
        - Periods: J.T. Realmuto, J.D. Davis, any “Jr.” 
        - Apostrophes: Tyler O’Neill, Travis d’Arnaud 
        - Hyphens: Pete Crow-Armstrong (but I did keep him in)
      - \> 3 tokens: e.g. Michael A. Taylor 
      - Robbie Ray :/ (see: Rays)
- Occasional challenges from lowercasing:
  - Word “era” vs. stat “ERA” (see also: WHIP, but ERA def a bigger problem)
- Didn't do smth like comparing player similarity against a similarity value b/c
  it feels nearly impossible to say which players are the most similar to each other
  in many cases. Bill James similarity score metric exists, but quality is 
  questionable + would take work to implement
- Hard to figure out ideal settings for PCA/t-SNE vis, clustering

## Acknowledgments
- marc ofc
- my dad especially for connecting me w/ resources
- but also the rest of my family (mom, jojo, deena, etc)
- sean forman
- rob arthur
- dan szymborski
- daniel
- the rest of my friends (ren, lola, joanna, shel, klo, etc)

## Bibliography

**Note:** Not all items listed below are referenced in the body of the above paper.
However, all were influential to some degree in the process of developing this 
project, and as such have been included to provide a comprehensive and fair 
overview of the literature I found useful.

<div class="csl-bib-body" style="line-height: 1.35; ">
  <div class="csl-entry" style="margin-bottom: 1em;">Abdulkareem Alsudais and Hovig Tchalian. 2019. <a href="https://doi.org/10.48550/arXiv.1807.10800">Clustering prominent named entities in topic-specific text corpora</a>. <i>Preprint</i>, arXiv:1807.10800v2.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Maria Antoniak and David Mimno. 2018. <a href="https://aclanthology.org/Q18-1008/">Evaluating the stability of embedding-based word similarities</a>. <i>Transactions of the Association for Computational Linguistics</i>, 6:107–119.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Zachary William Arth and Andrew C. Billings. 2019. <a href="https://doi.org/10.1080/10646175.2018.1466746">Touching racialized bases: Ethnicity in Major League Baseball broadcasts at the local and national levels</a>. <i>Howard Journal of Communications</i>, 30(3):230–248.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Robert Arthur. 2020a. <a href="https://www.baseballprospectus.com/news/article/59340/moonshot-baseball-ought-to-confront-its-own-racism/">Moonshot: Baseball ought to confront its own racism</a>. <i>Baseball Prospectus</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Robert Arthur. 2020b. <a href="https://www.baseballprospectus.com/news/article/59905/moonshot-racial-bias-shapes-which-players-make-the-majors/">Moonshot: Racial bias shapes which players make the majors</a>. <i>Baseball Prospectus</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Robert Arthur. 2026. <a href="https://www.baseballprospectus.com/news/article/105805/moonshot-prospect-hunting-with-deep-learning/">Prospect hunting with deep learning</a>. <i>Baseball Prospectus</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. 2017. <a href="https://www.science.org/doi/10.1126/science.aal4230">Semantics derived automatically from language corpora contain human-like biases</a>. <i>Science</i>, 356(6334):183–186.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Russell A. Carleton. 2012. <a href="https://www.baseballprospectus.com/news/article/18229/baseball-therapy-is-there-really-racism-in-the-broadcast-booth/">Baseball therapy: Is there really racism in the broadcast booth?</a>. <i>Baseball Prospectus</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Miriam Cha, Youngjune Gwon, and H. T. Kung. 2017. <a href="https://doi.org/10.48550/arXiv.1709.01888">Language modeling by clustering with word embeddings for text readability assessment</a>. <i>Preprint</i>, arXiv:1709.01888.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Jacob Danovitch. 2019. <a href="https://doi.org/10.48550/arXiv.1910.12622">Trouble with the curve: Predicting future MLB players using scouting reports</a>. <i>Preprint</i>, arXiv:1910.12622.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Andrea N. Eagleman. 2008. <a href="https://www.proquest.com/docview/288401138/abstract/4E0FB072D7034858PQ/1">Investigating agenda-setting and framing in sport magazines: An analysis of the coverage of Major League Baseball players from 2000 through 2007</a>. Ph.D. dissertation, Indiana University.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Adam Felder and Seth Amitin. 2025. <a href="https://www.theatlantic.com/entertainment/archive/2012/08/how-mlb-announcers-favor-american-players-over-foreign-ones/261265/">How MLB announcers favor American players over foreign ones</a>. <i>The Atlantic</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Patrick Ferrucci, Edson C. Tandoc, Jr., Chad E. Painter, and Glenn Leshner. 2013. <a href="https://doi.org/10.1080/10646175.2013.805971">A Black and White game: Racial stereotypes in baseball</a>. <i>Howard Journal of Communications</i>, 24(3):309–325.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Patrick Ferrucci, Edson C. Tandoc, Jr., Chad E. Painter, and J. David Wolfgang. 2016. <a href="https://doi.org/10.1080/10646175.2015.1117029">Foul ball: Audience-held stereotypes of baseball players</a>. <i>Howard Journal of Communications</i>, 27(1):68–84.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Matthew Francis-Landau, Greg Durrett, and Dan Klein. 2016. <a href="https://aclanthology.org/N16-1150/">Capturing semantic similarity for entity linking with convolutional neural networks</a>. In <i>Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies</i>, pages 1256–1261, San Diego, California. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Rian Emmerson Garcia. 2022. <a href="https://scholarworks.calstate.edu/downloads/3x816v36h">African American representation and racial bias in sports broadcast journalism</a>. M.A. thesis, California State University, Northridge.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Patrick C. Gentile. 2023. <a href="https://doi.org/10.1080/10646175.2022.2099771">"Learning English is the single most important thing": A qualitative analysis of the linguistic acquisition of Latino Minor League Baseball players</a>. <i>Howard Journal of Communications</i>, 34(2):113–131.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Patrick C. Gentile and Nicholas R. Buzzelli. 2021. <a href="https://doi.org/10.1080/10646175.2021.1879694">Hablamos inglés: Media portrayals of English-proficient Latin American MLB players</a>. <i>Howard Journal of Communications</i>, 32(3):253–273.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Lettie González, E. Newton Jackson, and Robert M. Regoli. 2006. <a href="https://www.jstor.org/stable/41819121">The transmission of racist ideology in sport: Using photo-elicitation to gauge success in professional baseball</a>. <i>Journal of African American Studies</i>, 10(3):46–54.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Johannes Hellrich and Udo Hahn. 2016. <a href="https://aclanthology.org/C16-1262/">Bad company — Neighborhoods in neural embedding spaces considered harmful</a>. In <i>Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers</i>, pages 2785–2796, Osaka, Japan. The COLING 2016 Organizing Committee.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Ezra Keshet, Terrence Szymanski, and Stephen Tyndall. 2011. <a href="https://aclanthology.org/W11-0139/">BALLGAME: A corpus for computational semantics</a>. In <i>Proceedings of the Ninth International Conference on Computational Semantics (IWCS 2011)</i>.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Omer Levy, Yoav Goldberg, and Ido Dagan. 2015. <a href="https://doi.org/10.1162/tacl_a_00134">Improving distributional similarity with lessons learned from word embeddings</a>. <i>Transactions of the Association for Computational Linguistics</i>, 3:211–225.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Ben Lindbergh and Robert Arthur. 2019. <a href="https://www.theringer.com/2019/03/04/mlb/cincinnati-reds-scouting-report-series-part-1-data-findings">We got our hands on 73,000 never-before-seen MLB scouting reports. Here's what we learned</a>. <i>The Ringer</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;"> Major League Baseball. 2025. <a href="https://www.mlb.com/press-release/press-release-opening-day-rosters-feature-265-internationally-born-players">Opening Day rosters feature 265 internationally born players</a>. <i>MLB.com</i>. Accessed May 6, 2026.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Jack Merullo, Luke Yeh, Abram Handler, Alvin Grissom II, Brendan O’Connor, and Mohit Iyyer. 2019. <a href="https://aclanthology.org/D19-1666/">Investigating sports commentator bias within a large corpus of American football broadcasts</a>. In <i>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)</i>, pages 6355–6361, Hong Kong, China. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. <a href="https://doi.org/10.48550/arXiv.1301.3781">Efficient estimation of word representations in vector space</a>. <i>Preprint</i>, arXiv:1301.3781.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Alice Oh and Howard Shrobe. 2008. <a href="https://aclanthology.org/W08-1124/">Generating baseball summaries from multiple perspectives by reordering content</a>. In <i>Proceedings of the Fifth International Natural Language Generation Conference</i>, pages 173–176, Salt Fork, Ohio, USA. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Matthew E. Peters, Mark Neumann, Robert Logan, Roy Schwartz, Vidur Joshi, Sameer Singh, and Noah A. Smith. 2019. <a href="https://aclanthology.org/D19-1005/">Knowledge enhanced contextual word representations</a>. In <i>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)</i>, pages 43–54, Hong Kong, China. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Nina Poerner, Ulli Waltinger, and Hinrich Schütze. 2020. <a href="https://aclanthology.org/2020.findings-emnlp.71/">E-BERT: Efficient-yet-effective entity embeddings for BERT</a>. In <i>Findings of the Association for Computational Linguistics: EMNLP 2020</i>, pages 803–818, Online. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Andrew Runge and Eduard Hovy. 2020. <a href="https://aclanthology.org/2020.blackboxnlp-1.20/">Exploring neural entity representations for semantic information</a>. In <i>Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP</i>, pages 204–216, Online. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Sports Reference LLC. 2025. <a href="https://www.baseball-reference.com/">Baseball-Reference.com - Major League Statistics and Information.</a>. Database downloaded December 29, 2025. </div>
  <div class="csl-entry" style="margin-bottom: 1em;">Yingtao Tian, Vivek Kulkarni, Bryan Perozzi, and Steven Skiena. 2016. <a href="https://doi.org/10.48550/arXiv.1605.03956">On the convergent properties of word embedding methods</a>. <i>Preprint</i>, arXiv:1605.03956.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Ikuya Yamada, Hiroyuki Shindo, Hideaki Takeda, and Yoshiyasu Takefuji. 2016. <a href="https://aclanthology.org/K16-1025/">Joint learning of the embedding of words and entities for named entity disambiguation</a>. In <i>Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning</i>, pages 250–259, Berlin, Germany. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Ikuya Yamada, Akari Asai, Jin Sakuma, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, and Yuji Matsumoto. 2020a. <a href="https://doi.org/10.48550/arXiv.1812.06280">Wikipedia2Vec: An efficient toolkit for learning and visualizing the embeddings of words and entities from Wikipedia</a>. <i>Preprint</i>, arXiv:1812.06280.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, and Yuji Matsumoto. 2020b. <a href="https://aclanthology.org/2020.emnlp-main.523/">LUKE: deep contextualized entity representations with entity-aware self-attention</a>. In <i>Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)</i>, pages 6442–6454, Online. Association for Computational Linguistics.</div>
  <div class="csl-entry" style="margin-bottom: 1em;">Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu. 2019. <a href="https://aclanthology.org/P19-1139/">ERNIE: Enhanced language representation with informative entities</a>. In <i>Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics</i>, pages 1441–1451, Florence, Italy. Association for Computational Linguistics.</div>
</div>