import json, random, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from adjustText import adjust_text
from gensim.models import Word2Vec
from utils import DATA_DIR, DATABASE, ARTICLE_STORAGE, PARSED_TEXT_STORAGE, CLEAN_TEXT_STORAGE, TKN_TEXT_STORAGE, MASK_TEXT_STORAGE, MENT_TEXT_STORAGE, RANDOM_SEED, ch_dir


def cluster_embeddings(embeds):
    n_clusters = 18
    metric = 'cosine' # euclidean, cosine, l1, l2, manhattan
    linkage = 'average' # single, complete, average, ward (must use euclidean)
    scores = {}

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)

    X, targets = get_target_array(embeds)
    km_labels = km.fit_predict(X)
    agg_labels = agg.fit_predict(X)

    km_cluster_members = {n: [] for n in range(n_clusters)}
    agg_cluster_members = {n: [] for n in range(n_clusters)}
    for i in range(len(targets)):
        km_cluster_members[km_labels[i]].append(targets[i])
        agg_cluster_members[agg_labels[i]].append(targets[i])
    with open(f'clusters/members_kmeans.json', 'w', encoding='utf-8') as f:
        json.dump(km_cluster_members, f, ensure_ascii=False, indent=4)
    print(f'K-means cluster members saved to {os.getcwd()}/clusters/members_kmeans.json.')
    with open(f'clusters/members_agg.json', 'w', encoding='utf-8') as f:
        json.dump(agg_cluster_members, f, ensure_ascii=False, indent=4)
    print(f'Agglomerative cluster members saved to {os.getcwd()}/clusters/members_agg.json.')

    km_score = silhouette_score(X, km_labels, metric='cosine')
    scores['K-Means'] = km_score
    agg_score = silhouette_score(X, agg_labels, metric='cosine')
    scores[f'Agglomerative ({metric} + {linkage})'] = agg_score

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14), layout='tight')
    ax1.set_xlim([-0.4, 0.6])
    ax1.set_ylim([0, len(X) + 1])
    ax3.set_xlim([-0.4, 0.6])
    ax3.set_ylim([0, len(X) + 1])

    km_samples = silhouette_samples(X, km_labels, metric='cosine')
    agg_samples = silhouette_samples(X, agg_labels, metric='cosine')

    vd = plt.get_cmap('viridis', n_clusters)
    y_lower = 1
    for i in range(n_clusters):
        km_i = km_samples[km_labels == i]
        km_i.sort()
        km_size_i = len(km_i)
        y_upper = y_lower + km_size_i

        color = vd(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            km_i,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        y_lower = y_upper

    pl = plt.get_cmap('plasma', n_clusters)
    y_lower = 1
    for i in range(n_clusters):
        agg_i = agg_samples[agg_labels == i]
        agg_i.sort()
        agg_size_i = len(agg_i)
        y_upper = y_lower + agg_size_i

        color = pl(float(i) / n_clusters)
        ax3.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            agg_i,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        y_lower = y_upper

    ax1.set_xlabel('Silhouette coefficient values')
    ax1.axvline(x=km_score, color='red', linestyle='--')
    ax3.set_xlabel('Silhouette coefficient values')
    ax3.axvline(x=agg_score, color='red', linestyle='--')

    ax1.set_yticks([])
    ax1.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax3.set_yticks([])
    ax3.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    x_vals, y_vals = reduce_dimensions(X)

    colors = vd(km_labels.astype(float) / n_clusters)
    km_points = ax2.scatter(
        x_vals, y_vals, alpha=0.7, c=colors
    )

    colors = pl(agg_labels.astype(float) / n_clusters)
    agg_points = ax4.scatter(
        x_vals, y_vals, alpha=0.7, c=colors
    )

    ax2.set_xlabel('Feature space for the 1st feature')
    ax2.set_ylabel('Feature space for the 2nd feature')
    ax4.set_xlabel('Feature space for the 1st feature')
    ax4.set_ylabel('Feature space for the 2nd feature')

    km_texts = [ax2.text(x_vals[i], y_vals[i], targets[i], ha='center', va='center', size='small') for i in
             range(len(targets))]
    agg_texts = [ax4.text(x_vals[i], y_vals[i], targets[i], ha='center', va='center', size='small') for i in
             range(len(targets))]

    adjust_text(
        km_texts,
        objects=km_points,
        # force_pull=(0.2, 0.3),
        # force_explode=(0.2, 0.5),
        # force_text=(0.3, 0.3),
        # force_static=(0.3, 0.2),
        # expand=(1, 1),
        # max_move=None,
        # pull_threshold=1,
        ax=ax2,
        time_lim=3,
        arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
        min_arrow_len=5,
    )
    adjust_text(
        agg_texts,
        objects=agg_points,
        # force_pull=(0.2, 0.3),
        # force_explode=(0.2, 0.5),
        # force_text=(0.3, 0.3),
        # force_static=(0.3, 0.2),
        # expand=(1, 1),
        # max_move=None,
        # pull_threshold=1,
        ax=ax4,
        time_lim=3,
        arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
        min_arrow_len=5,
    )

    plt.suptitle(
        f'Silhouette analysis for K-Means and agglomerative clustering with # clusters = {n_clusters}',
        fontsize=14,
        fontweight='bold',
    )

    print(scores)
    fig.savefig(f'figs/clusters.png')
    print(f'Plots for all clusters saved to {os.getcwd()}/figs/clusters.png.')
    plt.show()


def evaluate_cluster_size(embeds):
    X, targets = get_target_array(embeds)
    range_n_clusters = [15, 18]

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout='tight')

        ax1.set_xlim([-0.4, 0.6])
        ax1.set_ylim([0, len(X) + 1])

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X)
        silco = silhouette_score(X, labels, metric='cosine') # euclidean, cosine, l1, l2, manhattan
        print(f'# clusters = {n_clusters}: {silco}')

        cluster_members = {n: [] for n in range(n_clusters)}
        for i in range(len(targets)):
            cluster_members[labels[i]].append(targets[i])
        with open(f'clusters/members_cluster_num_{n_clusters}.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_members, f, ensure_ascii=False, indent=4)
        print(f'Members for {n_clusters} clusters saved to {os.getcwd()}/clusters/members_cluster_num_{n_clusters}.json.')

        sample_silco = silhouette_samples(X, labels, metric='cosine')

        vd = plt.get_cmap('plasma', n_clusters)
        y_lower = 1
        for i in range(n_clusters):
            ith_cluster_silco = sample_silco[labels == i]
            ith_cluster_silco.sort()
            size_cluster_i = len(ith_cluster_silco)
            y_upper = y_lower + size_cluster_i

            color = vd(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silco,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # ax1.text(-0.02, (y_upper - y_lower) / 2, str(i))
            y_lower = y_upper

        ax1.set_xlabel('Silhouette coefficient values')
        ax1.axvline(x=silco, color='red', linestyle='--')

        ax1.set_yticks([])
        ax1.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        colors = vd(labels.astype(float) / n_clusters)
        x_vals, y_vals = reduce_dimensions(X)
        points = ax2.scatter(
            x_vals, y_vals, alpha=0.7, c=colors
        )

        ax2.set_xlabel('Feature space for the 1st feature')
        ax2.set_ylabel('Feature space for the 2nd feature')

        texts = [plt.text(x_vals[i], y_vals[i], targets[i], ha='center', va='center', size='small') for i in range(len(targets))]
        adjust_text(
            texts,
            objects=points,
            # force_pull=(0.2, 0.3),
            # force_explode=(0.2, 0.5),
            # force_text=(0.3, 0.3),
            # force_static=(0.3, 0.2),
            # expand=(1, 1),
            # max_move=None,
            # pull_threshold=1,
            ax=ax2,
            time_lim=3,
            arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
            min_arrow_len=5,
        )

        plt.suptitle(
            f'Silhouette analysis for K-Means clustering with # clusters = {n_clusters}',
            fontsize=14,
            fontweight='bold',
        )

        fig.savefig(f'figs/test_cluster_num_{n_clusters}.png')
        print(f'Plots for {n_clusters} clusters saved to {os.getcwd()}/figs/test_cluster_num_{n_clusters}.png.')

    plt.show()


def compare_linkage_metrics(embeds):
    X, targets = get_target_array(embeds)
    metrics = ['euclidean', 'cosine', 'l1', 'l2', 'manhattan']
    linkages = ['single', 'complete', 'average', 'ward']
    n_clusters = 6

    for link in linkages:
        for m in metrics:
            if link == 'ward' and m != 'euclidean':
                continue
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout='tight')

            ax1.set_xlim([-0.3, 0.6])
            ax1.set_ylim([0, len(X) + 1])

            agg = AgglomerativeClustering(n_clusters=n_clusters, metric=m, linkage=link)
            labels = agg.fit_predict(X)
            silco = silhouette_score(X, labels, metric=m)
            print(f'{link} linkage, {m} distance, {n_clusters} clusters: {silco}')

            sample_silco = silhouette_samples(X, labels, metric=m)

            vd = plt.get_cmap('viridis', n_clusters)
            y_lower = 1
            for i in range(n_clusters):
                ith_cluster_silco = sample_silco[labels == i]
                ith_cluster_silco.sort()
                size_cluster_i = len(ith_cluster_silco)
                y_upper = y_lower + size_cluster_i

                color = vd(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silco,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
                # ax1.text(-0.02, (y_upper - y_lower) / 2, str(i))
                y_lower = y_upper

            ax1.set_xlabel('Silhouette coefficient values')
            ax1.axvline(x=silco, color='red', linestyle='--')

            ax1.set_yticks([])
            ax1.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            colors = vd(labels.astype(float) / n_clusters)
            x_vals, y_vals = reduce_dimensions(X)
            points = ax2.scatter(
                x_vals, y_vals, alpha=0.7, c=colors
            )

            ax2.set_xlabel('Feature space for the 1st feature')
            ax2.set_ylabel('Feature space for the 2nd feature')

            texts = [plt.text(x_vals[i], y_vals[i], targets[i], ha='center', va='center', size='small') for i in
                     range(len(targets))]
            adjust_text(
                texts,
                objects=points,
                # force_pull=(0.2, 0.3),
                # force_explode=(0.2, 0.5),
                # force_text=(0.3, 0.3),
                # force_static=(0.3, 0.2),
                # expand=(1, 1),
                # max_move=None,
                # pull_threshold=1,
                ax=ax2,
                time_lim=3,
                arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                min_arrow_len=5,
            )

            plt.suptitle(
                f'Silhouette analysis for agglomerative clustering with {link} linkage, {m} distance, {n_clusters} clusters',
                fontsize=12,
                fontweight='bold',
            )

            fig.savefig(f'figs/test_{link}_{m}.png')
            print(f'Plots for {link} linkage, {m} distance, {n_clusters} clusters saved to {os.getcwd()}/figs/test_{link}_{m}.png.')

    plt.show()


def get_target_array(embeds):
    with open('player_list.txt', 'r', encoding='utf-8') as f:
        players = [line.strip() for line in f]
    vectors = np.array([embeds.get_vector(p) for p in players])
    targets = players

    # with open('data/target_words.json', 'r', encoding='utf-8') as f:
    #     words = json.load(f)
    # words = [w for label in words for w in words[label]]
    # targets = players + words
    # vectors = np.array([embeds.get_vector(t) for t in targets])

    return vectors, targets


def reduce_dimensions(vectors):
    dims = 2

    pca = IncrementalPCA(n_components=dims)
    vectors = pca.fit_transform(vectors)
    # tsne = TSNE(n_components=dims, perplexity=(len(vectors) / 4), max_iter=5000, random_state=RANDOM_SEED)
    # vectors = tsne.fit_transform(vectors)

    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]
    return x_vals, y_vals


def plot_embeds(embeds):
    vectors, labels = get_target_array(embeds)
    x_vals, y_vals = reduce_dimensions(vectors)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    texts = [plt.text(x_vals[i], y_vals[i], labels[i]) for i in range(len(labels))]
    adjust_text(texts)
    # for i in range(len(labels)):
    #     plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()
    # plt.savefig(output_file)
    # plt.close()


def get_cluster_stats(cluster_path):
    with open(cluster_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)

    stats = {'person_id': [], 'cluster': []}
    for k, v in clusters.items():
        for pid in v:
            stats['person_id'].append(pid)
            stats['cluster'].append(int(k) + 1)
    clus = pd.DataFrame.from_dict(stats)
    nats = pd.read_csv('player_nationality.tsv', sep='\t', names=['person_id', 'country'])
    batter_df = pd.read_csv(f'stats/batter_stats_calc.csv', encoding='utf-8', usecols=[0, 1, 3])
    pitcher_df = pd.read_csv(f'stats/pitcher_stats_calc.csv', encoding='utf-8', usecols=[0, 1, 3])
    player_df = pd.concat([batter_df, pitcher_df])

    merged_df = pd.merge(clus, nats, how='left', on='person_id')
    merged_df = pd.merge(merged_df, player_df, how='left', on='person_id')

    fname = Path(cluster_path).stem
    merged_df.to_csv(f'clusters/{fname}_stats.csv', index=False, mode='w', encoding='utf-8')


def plot_cluster_stats(fpath):
    df = pd.read_csv(fpath, encoding='utf-8')
    for i in range(18):
        df_rows = df[df['cluster'] == i + 1]
    clusters = (i + 1 for i in range(18))
    penguin_means = {
        'Team': (18.35, 18.43, 14.98),
        'Position': (38.79, 48.83, 47.50),
        'Nationality': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    species = (
        "Adelie\n $\\mu=$3700.66g",
        "Chinstrap\n $\\mu=$3733.09g",
        "Gentoo\n $\\mu=5076.02g$",
    )
    weight_counts = {
        "Below": np.array([70, 31, 58]),
        "Above": np.array([82, 37, 66]),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(3)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.show()


if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    exp_num = '005'
    exp_path = f'data/exp_{exp_num}'

    with ch_dir(exp_path):
        os.makedirs('figs', exist_ok=True)
        os.makedirs('clusters', exist_ok=True)
        # for model_path in glob.glob('*.model'):
        #     model = Word2Vec.load(model_path)
        #     # evaluate_cluster_size(model.wv)
        #     # compare_linkage_metrics(model.wv)
        #     cluster_embeddings(model.wv)
        for cluster_path in glob.glob('clusters/*.json'):
            get_cluster_stats(cluster_path)