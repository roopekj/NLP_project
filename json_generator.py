import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer

ZOOM_LEVELS = [5, 10, 20, -1] # -1 is for the full graph (clusters with the second to last zoom level)
MODEL = 'all-MiniLM-L6-v2'
OUTPUT_FILE = 'app/data/data.json'

def generate_clusters(embeddings: np.ndarray, n_clusters: int) -> list[int]:
    """
    Generate clusters for the given embeddings.

    Parameters
    ----------
    embeddings -> the embeddings to cluster.
    zoom_level -> the zoom level of the graph.

    Returns
    -------
    The clusters for the given embeddings.
    """

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(embeddings)
    clusters = kmeans.labels_

    return clusters.tolist()

def generate_tsne_embeddings(embeddings: np.ndarray, zoom_level: int, clusters: list[int]) -> dict[int, list[float]]:
    """
    Generate t-SNE embeddings for the given embeddings.

    Parameters
    ----------
    embeddings -> the embeddings to generate t-SNE embeddings for.
    zoom_level -> the zoom level of the graph.

    Returns
    -------
    The t-SNE embeddings for the given embeddings.
    """

    # Generate t-SNE embeddings
    tsne = TSNE(n_components=2, n_iter=1000, n_iter_without_progress=200, perplexity=35)
    tsne_embeddings = tsne.fit_transform(embeddings)

    if zoom_level == -1:
        return tsne_embeddings.tolist()
    else:
        # return one embedding per cluster centered around the mean of the cluster
        cluster_embeddings = []
        for i in range(max(clusters) + 1):
            tmp = list((np.mean(tsne_embeddings[[j for j in range(len(tsne_embeddings)) if clusters[j] == i]], axis=0)))
            cluster_embeddings.append([float(x) for x in tmp])
        return cluster_embeddings


def generate_cluster_keywords(newsgroups_train: dict, labels: list[int] ,n_clusters: int) -> list[list[str]]:
    """
    Generate cluster keywords for the given clusters.

    Parameters
    ----------
    newsgroups_train -> the training data.
    n_clusters -> the number of clusters.

    Returns
    -------
    The cluster keywords for the given clusters.
    """
    cluster_keywords = []

    for i in range(n_clusters):
        cluster_data = [newsgroups_train.data[j] for j in range(len(newsgroups_train.data)) if labels[j] == i]
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(cluster_data)
        lda = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)
        kwds = []
        for topic_idx, topic in enumerate(lda.components_):
            feature_names = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1] if vectorizer.get_feature_names_out()[i] not in kwds and not vectorizer.get_feature_names_out()[i].isdigit()]
            kwds.extend(feature_names)
        cluster_keywords.append(kwds)

    return {i: cluster_keywords[i] for i in range(len(cluster_keywords))}


def main():
    global ZOOM_LEVELS, MODEL, OUTPUT_FILE
    parser = argparse.ArgumentParser(description='Generate JSON data for the 20 Newsgroups dataset.')
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_FILE, help='Output file path')
    parser.add_argument('-m', '--model', type=str, default=MODEL, help='SBERT model to use')
    parser.add_argument('-z', '--zoom', type=int, nargs='+', default=ZOOM_LEVELS, help='Zoom levels to use')
    args = parser.parse_args()

    ZOOM_LEVELS = args.zoom
    MODEL = args.model
    OUTPUT_FILE = args.output
    print(f'Using SBERT model {MODEL} and zoom levels {ZOOM_LEVELS}...')

    print('Fetching 20 newsgroups dataset...')
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    idxs = [ i for i in range(len(newsgroups_train.data)) if len(newsgroups_train.data[i]) > 20 ]
    newsgroups_train.data = [ newsgroups_train.data[i] for i in idxs ]
    newsgroups_train.target = [ newsgroups_train.target[i] for i in idxs ]

    json_data = {}
    json_data['dataset'] = {
        'target_names': newsgroups_train.target_names,
        'samples': [ {
            'data': newsgroups_train.data[i],
            'target': int(newsgroups_train.target[i])
        } for i in range(len(newsgroups_train.data)) ]
    }
    json_data['zoom_levels'] = ZOOM_LEVELS

    model = SentenceTransformer(MODEL)
    print('Generating SBERT embeddings...')
    train_embeddings = model.encode(newsgroups_train.data, show_progress_bar=True)
    train_embeddings.shape

    json_data['data'] = {}

    for zoom_level in ZOOM_LEVELS:
        if zoom_level != -1:
            data_dict = {}
            print(f'Generating clusters for zoom level {zoom_level}...')
            clusters = generate_clusters(train_embeddings, zoom_level)
            data_dict['clusters'] = list(range(max(clusters) + 1))

            print(f'Generating t-SNE embeddings for zoom level {zoom_level}...')
            tsne_embeddings = generate_tsne_embeddings(train_embeddings, zoom_level, clusters)
            data_dict['tsne_embeddings'] = tsne_embeddings
            

            print(f'Generating cluster keywords for zoom level {zoom_level}...')
            cluster_keywords = generate_cluster_keywords(newsgroups_train, clusters, zoom_level)
            data_dict['cluster_keywords'] = cluster_keywords

            json_data['data'][zoom_level] = data_dict

        else:
            data_dict = {}
            zoom_level = max(ZOOM_LEVELS) if max(ZOOM_LEVELS) > 0 else 1
            clusters = generate_clusters(train_embeddings, zoom_level)
            data_dict['clusters'] = clusters

            print(f'Generating t-SNE embeddings for full zoom level...')
            tsne_embeddings = generate_tsne_embeddings(train_embeddings, -1, clusters)
            data_dict['tsne_embeddings'] = tsne_embeddings
            
            print(f'Generating cluster keywords for full zoom level...')
            cluster_keywords = generate_cluster_keywords(newsgroups_train, clusters, zoom_level)
            data_dict['cluster_keywords'] = cluster_keywords

            json_data['data'][-1] = data_dict

    print(f'Writing JSON data to {OUTPUT_FILE}...')
    with open(OUTPUT_FILE, 'w+') as outfile:
        json.dump(json_data, outfile)

if __name__ == '__main__':
    main()