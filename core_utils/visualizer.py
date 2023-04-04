"""
Visualizer module for visualizing PosFrequencyPipeline results
"""
from pathlib import Path
from typing import Callable

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np  # type: ignore

    matplotlib.use('agg')
except ImportError:  # pragma: no cover
    print('No libraries installed. Failed to import.')

from core_utils.article.article import Article


def visualize(article: Article, path_to_save: Path) -> None:
    """
    param: article is an instance of the Article class
    """
    statistics = article.get_pos_freq()
    number_of_tags = len(statistics)
    sorted_frequencies = sorted(statistics.values(), reverse=True)
    get_occurrences: Callable = lambda x: statistics[x]
    sorted_tags = sorted(statistics, key=get_occurrences, reverse=True)
    pos_tags = np.arange(number_of_tags)
    colors = ('b', 'g', 'r', 'c')

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    for i in range(0, number_of_tags):
        axis.bar(pos_tags[i], sorted_frequencies[i],
                 align='center', width=0.5,
                 color=colors[i % len(colors)])

    axis.set_xticks(pos_tags)
    axis.set_xticklabels(sorted_tags)
    plt.setp(sorted_tags)
    plt.xticks(rotation=20)
    y_max = max(sorted_frequencies) + 1
    plt.ylim(0, y_max)

    plt.savefig(path_to_save)
