import numpy as np, pickle, operator
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


class ImageClustering(object):

    def __init__(self, path_to_features, number_of_clusters=10, compression_percentage=0.8):
        # data structures
        self.path_to_features = path_to_features
        self.number_of_clusters = number_of_clusters
        self.compression_percentage = compression_percentage
        self.feature_vectors = self.compute_feature_vectors()
        self.data_set = self.prepare_data()
        self.connectivity_matrix = self.compute_connectivity_matrix()
        # this is vector full of clusters containing file names and a dictionary where key is filename e.g. '000001.txt' and value is number of the cluster
        self.clusters, self.reversed_clusters = self.compute_clusters()
        self.centroids = self.compute_centroids()

    # This is preparing the data so that the clustering algorithm can use it
    def compute_feature_vectors(self):
        # todo build numpy arrays for feature vectors or load from some other class
        filename = ''
        feature_vectors = pickle.load(open(self.path_to_features + filename, "rb"))
        return feature_vectors

    # This is preparing the data so that the clustering algorithm can use it
    def prepare_data(self):
        features_transformed = []
        for x, y in self.feature_vectors.items():
            features_transformed.append(y)

        data_set = np.stack(features_transformed, axis=0)

        # PCA will decrease the number of dimensions of our data, it will decrease it as much as possible but will keep 75% of the information from the data
        pca = PCA(n_components=self.compression_percentage, svd_solver='full')
        pca.fit(data_set)
        data_set = pca.transform(data_set)

        # todo try to pickle the data and load it for efficiency
        # pickle.dump(data_set, open("data_set.p", "wb"))
        # data_set = pickle.load(open("data_set.p", "rb"))
        return data_set

    # This builds a connectivity matrix and can be used to improve AgglomerativeClustering Performance by a lot
    def compute_connectivity_matrix(self):
        # todo try to pickle the data and load it for efficiency
        connectivity_matrix = kneighbors_graph(self.data_set, n_neighbors=20, include_self=False)
        # connectivity_matrix = pickle.load(open("connectivityMatrix.p", "rb"))
        # pickle.dump(connectivity_matrix, open("connectivityMatrix.p", "wb"))
        return connectivity_matrix

    def compute_clusters(self):
        clustering = AgglomerativeClustering(n_clusters=self.number_of_clusters, connectivity=self.connectivity_matrix,
                                             linkage='ward').fit(
            self.data_set)

        # todo try to pickle the data and load it for efficiency
        # clustering = pickle.load(open("clustering.p", "rb"))
        # pickle.dump(clustering, open("clustering.p", "wb"))

        clusterLabels = clustering.labels_

        images_labels = []
        for x, y in self.feature_vectors.items():
            images_labels.append(x)

        clusters = []
        reversed_clusters = {}
        for x in range(0, self.number_of_clusters):
            clusters.append([])
        n = 0
        for x in clusterLabels:
            filename = images_labels[n]
            reversed_clusters[filename] = x
            clusters[x].append(filename)
            n = n + 1
        return clusters, reversed_clusters

    # this method computes the centroids for all clusters
    def compute_centroids(self):
        feature_vectors = self.feature_vectors
        clusters = self.clusters
        centroids = []
        for x in clusters:
            feature_vectors_in_cluster = []
            for y in x:
                feature_vectors_in_cluster.append(feature_vectors[y])
            dimensions = feature_vectors['002492.txt'].size
            centroid = np.zeros(dimensions)
            for z in feature_vectors_in_cluster:
                centroid = np.add(centroid, z)
            n = len(feature_vectors_in_cluster)
            centroid = np.divide(centroid, n)
            centroids.append(centroid)
        return centroids

    # this method will find the k closest neighbour clusters for a given cluster
    def findKNeighboursCentroids(self, k, index_of_centroid):
        # building initial array for 5 neighbours
        neighbours = []
        centroid = self.centroids[index_of_centroid]
        for x in range(0, k):
            neighbours.append((float('inf'), '0'))
        n = 0
        for centroid_temp in self.centroids:
            distance = self.calculateEuclideanDistance(centroid, centroid_temp)
            if (distance < neighbours[k - 1][0]):
                neighbours[k - 1] = (distance, n)
                neighbours.sort(key=operator.itemgetter(0))
            n = n + 1
        return neighbours

    # this function calculates the euclidean distance between 2 feature vectors
    @staticmethod
    def compute_euclidean_distance(features1, features2):
        return np.linalg.norm(features1 - features2)

if __name__ == "__main__":
    obj = ImageClustering('/Users/volk/Desktop/Forschungspraktikum/vectors10000.p', 10, 0.8)
    print(obj.clusters)
    print(obj.reversed_clusters)