import numpy as np
from operator import itemgetter

class KNearestNeighbour:

    def __init__(self, data, target, test_point, k):
        self.data = data
        self.target = target
        self.test_point = test_point
        self.k = k
        self.distances = list()
        self.categories = list()
        self.indices = list()
        self.counts = list()
        self.category_assigned = None

    @staticmethod
    def dist(p1, p2):

        """ method returns the euclidean distance between two points """
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def fit(self):
        """ Method for performing KNN Classifications """

        # create a list of (distance, index) tuples from the test point to each point in data
        self.distances.extend([(self.dist(self.test_point, point), i) for point, i in zip(self.data, [i for i in range(len(self.data))])])

        # sorting the distances in ascending order
        sorted_li = sorted(self.distances, key=itemgetter(0))

        # fetching the indices of the K nearest point from the data
        self.indices.extend([index for (val, index) in sorted_li[:self.k]])

        # fetching the categories from train data target
        for i in self.indices:
            self.categories.append(self.target[i])

        # Fetch the count for each category from the K nearest neighbours
        self.counts.extend([(i, self.categories.count(i)) for i in set(self.categories)])

        # Find the highest repeated category among the K nearest neighbours
        self.category_assigned = sorted(self.counts, key=itemgetter(1), reverse=True)[0][0]



