from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import umap

plt.rcParams['figure.figsize'] = (10, 5)

def save_topic_visualization(embeddings, labels, output_path):
    """ Save topics visualization to a given output path 
    
    :param embeddings: sentence embeddings
    :param labels: cluster labels
    :param output_path: path to save visualization
    """
    # visualize clusters
    umap_data = umap.UMAP(n_neighbors=15,
                        n_components=2,
                        min_dist=0.0, 
                        metric='cosine',
                        random_state=42).fit_transform(embeddings)
    
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = labels

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.figure()
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.savefig(output_path)
    plt.clf()

def save_topic_wordclouds(topics, num_wordcloud_words, output_path):
    """ Save topics visualization to a given output path 
    
    :param topics: dictionary (topic:top_n_words)
    :param num_wordcloud_words: generated dictionary
    :param output_path: path to save visualization
    """

    for topic, top_n_words in topics.items():
        top_n_words = sorted(top_n_words, key=lambda x: x[1], reverse=True)[:num_wordcloud_words]
        dictionary = dict()
        for word, c_tf_idf in top_n_words:
            dictionary[word] = c_tf_idf
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(dictionary) 
        plt.figure()
        plt.axis('off')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.savefig(output_path + "/topic_" + str(topic))
        plt.clf()
    