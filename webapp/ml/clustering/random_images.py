import numpy as np, random, math
from PIL import Image
import time

from webapp.ml.clustering.image_clustering import ImageClustering


class RandomImages(object):

    def __init__(self, path_to_features, query_keyword, number_of_clusters=10):
        # data structures
        self.path_to_features = path_to_features
        self.query_keyword = query_keyword
        # self.number_of_clusters = number_of_clusters
        # todo try to pickle the data and load it for efficiency
        self.clustering = ImageClustering(path_to_features, number_of_clusters)
        self.probabilities, self.probabilities_accumalated = self.compute_probabilties()


    # This is preparing the data so that the clustering algorithm can use it
    def get_n_random_images(self, n):
        random_images = []
        for x in range(0, n):
            filename = random.choice(self.clustering.clusters[x % self.clustering.number_of_clusters])
            random_images.append(filename)
        return random_images

    # draws n samples from a given probability distribution
    def get_n_random_images_based_on_feedback(self, n):
        random_images = []
        for x in range(0, n):
            random_cluster = random.uniform(0, 1)
            for index_of_cluster in range(0, len(self.probabilities_accumalated)):
                if self.probabilities_accumalated[index_of_cluster] > random_cluster:
                    random_cluster = index_of_cluster
                    break
            filename = random.choice(self.clustering.clusters[random_cluster])
            random_images.append(filename)
        return random_images



    # adjusts the Probablities based on feedback
    def compute_probabilties(self):
        # 10 % der Bilder komplett Random, 70% basiert auf relevanten Bildern, 20% basiert auf irrelevanten
        # 0.1/number_of_clusters + (number_of_relevant_in_cluster-number_of_irrelevant_in_cluster)/number_of_relevant_images
        # todo decide how to store relevant and irrelevant images
        relevant_images = ['043785.txt', '012723.txt']
        irrelevant_images = ['058987.txt', '014541.txt', '045068.txt', '080452.txt']
        relevant_part = 0.7
        if len(relevant_images) == 0:
            relevant_part = 0
        irrelevant_part = 0.9 - relevant_part
        if len(irrelevant_images) == 0:
            irrelevant_part = 0
        random_part = 1 - irrelevant_part - relevant_part


        probabilities = []
        relevant_part_probablities = []
        irrelevant_part_probabilities = []
        for index_of_cluster in range(0, self.clustering.number_of_clusters):
            counter_relevant = 0
            for image in relevant_images:
                if self.clustering.reversed_clusters[image] == index_of_cluster:
                    counter_relevant += 1
            counter_irrelevant = 0
            for image in irrelevant_images:
                if self.clustering.reversed_clusters[image] == index_of_cluster:
                    counter_irrelevant += 1
            if len(relevant_images) != 0:
                relevant_part_probablities.append(relevant_part * counter_relevant/len(relevant_images))
                # relevant_part_temp = relevant_part * counter_relevant/len(relevant_images)
            if len(irrelevant_images) != 0:
                irrelevant_part_temp = 1/(math.pow(10, counter_irrelevant))
                # irrelevant_part_temp = (irrelevant_part / self.clustering.number_of_clusters)
                irrelevant_part_probabilities.append(irrelevant_part_temp)
            # probability = relevant_part_temp + irrelevant_part_temp + random_part/self.clustering.number_of_clusters
        sum = 0
        for irrelevant_part_temp in irrelevant_part_probabilities:
            sum += irrelevant_part_temp
        for index_of_cluster in range(0, self.clustering.number_of_clusters):
            probability = relevant_part_probablities[index_of_cluster] + (irrelevant_part*(irrelevant_part_probabilities[index_of_cluster]/sum)) + random_part/self.clustering.number_of_clusters
            probabilities.append(probability)


        probabilities_accumalated = []
        probabilities_accumalated.append(probabilities[0])
        # print(probabilities[0])
        for x in range(1, self.clustering.number_of_clusters):
            probabilities_accumalated.append(probabilities_accumalated[(x - 1)] + probabilities[x])
        return probabilities, probabilities_accumalated







if __name__ == "__main__":
    obj = RandomImages('/Users/volk/Desktop/Forschungspraktikum/vectors10000.p', 'whatever', 10)
    print(obj.clustering.clusters)
    print(obj.get_n_random_images(20))
    # obj = ImageClustering('/Users/volk/Desktop/Forschungspraktikum/vectors10000.p', 10, 0.8)
    print(obj.probabilities)
    print(obj.probabilities_accumalated)
    random_images = obj.get_n_random_images_based_on_feedback(20)