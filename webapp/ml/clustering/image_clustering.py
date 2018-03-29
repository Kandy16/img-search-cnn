import numpy as np, pickle, operator, os
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import pathlib2

class ImageClustering(object):

    def __init__(self, number_of_clusters=10, compression_percentage=0.8):
        # data structures
        self.number_of_clusters = number_of_clusters
        self.compression_percentage = compression_percentage
        #self.centroids = self.compute_centroids()

    def get_clusters(self, vectors_save_location , clusters_save_location , modelname , layername ):
        save_location = os.path.join(clusters_save_location , modelname, layername)
        cluster_file_location = os.path.join(save_location , 'clusters' + str(self.number_of_clusters) + '.p')
        reversed_clusters_location = os.path.join(save_location + 'reversed_clusters' + str(self.number_of_clusters) + '.p')
        if os.path.exists(cluster_file_location) and os.path.exists(reversed_clusters_location):
            print("clusters.p and reversed_clusters.p available at : " , os.path.join(save_location))                
            clusters = pickle.load(open(cluster_file_location, "rb"))
            reversed_clusters = pickle.load(open(reversed_clusters_location, "rb"))
            return clusters, reversed_clusters
        else:
            pathlib2.Path(os.path.join(clusters_save_location, modelname , layername)).mkdir(parents=True, exist_ok=True)
            return self._prepare_clusters(vectors_save_location , clusters_save_location , modelname , layername)


    # Returns vector full of clusters containing file names and a dictionary where key is filename e.g. '000001.txt' and value is number of the cluster
    def _prepare_clusters(self , vectors_save_location , clusters_save_location , modelname , layername):
        # First we check if vectors.p exists for given model and layer. Note we call KNN class for doing this as it is done there.
        save_location = os.path.join(clusters_save_location , modelname, layername)
        vectors_p_file_location = os.path.join(vectors_save_location, modelname , layername , "vectors.p")
        if os.path.exists(vectors_p_file_location):
            vectors_p = {}
            vectors_p = pickle.load(open(vectors_p_file_location , "rb"))
            data_set = self._prepare_data_set(vectors_p , save_location)
            connectivity_matrix = self._compute_connectivity_matrix(data_set , save_location)

            if os.path.exists(os.path.join(save_location , 'clusters' + str(self.number_of_clusters) + '.p')) and os.path.exists(os.path.join(save_location + 'reversed_clusters' + str(self.number_of_clusters) + '.p')):
                print("clusters.p and reversed_clusters.p available at : " , os.path.join(save_location))
                clusters = pickle.load(open(os.path.join(save_location , 'clusters' + str(self.number_of_clusters) + '.p'), "rb"))
                reversed_clusters = pickle.load(open(os.path.join(save_location + 'reversed_clusters' + str(self.number_of_clusters) + '.p'), "rb"))
            else:
                clustering = AgglomerativeClustering(n_clusters=self.number_of_clusters, connectivity= connectivity_matrix,
                                                     linkage='ward').fit(data_set)

                clusterLabels = clustering.labels_

                images_labels = []
                for x, y in vectors_p.items():
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
                    print(n)
                pickle.dump(clusters, open(os.path.join(save_location , 'clusters' + str(self.number_of_clusters) + '.p'), "wb"))
                pickle.dump(reversed_clusters, open(os.path.join(save_location , 'reversed_clusters' + str(self.number_of_clusters) + '.p'), "wb"))
            return clusters, reversed_clusters
        else:
            print('\n' + "vectors.p" + ' is not available at this location ' + vectors_p_file_location)
            exit()
            

    # This is preparing the data so that the clustering algorithm can use it
    def _prepare_data_set(self ,vectors_p,  save_location):
        if os.path.exists(os.path.join(save_location , "data_set.p")):
            print("data_set.p available at : " , save_location )
            data_set = pickle.load(open(os.path.join(save_location , "data_set.p"), "rb"))
        else:
            features_transformed = []
            for x, y in vectors_p.items():
                features_transformed.append(y)

            data_set = np.stack(features_transformed, axis=0)

            # PCA will decrease the number of dimensions of our data, it will decrease it as much as possible but will keep 75% of the information from the data
            pca = PCA(n_components=self.compression_percentage, svd_solver='full')
            pca.fit(data_set)
            data_set = pca.transform(data_set)
            pickle.dump(data_set, open(os.path.join(save_location , "data_set.p"), "wb"))
        return data_set


    # This builds a connectivity matrix and can be used to improve AgglomerativeClustering Performance by a lot
    def _compute_connectivity_matrix(self , data_set, save_location):
        if os.path.exists(os.path.join(save_location , 'connectivity_matrix.p')):
            print("connectivity_matrix.p is available at " , save_location)
            connectivity_matrix = pickle.load(open(os.path.join(save_location , 'connectivity_matrix.p'), "rb"))
        else:
            connectivity_matrix = kneighbors_graph(data_set, n_neighbors=20, include_self=False)
            pickle.dump(connectivity_matrix, open(os.path.join(save_location , 'connectivity_matrix.p'), "wb"))
        return connectivity_matrix

    # this method computes the centroids for all clusters
    def compute_centroids(self):
        feature_vectors = self.feature_vectors
        clusters = self.clusters
        centroids = []
        for x in clusters:
            feature_vectors_in_cluster = []
            for y in x:
                feature_vectors_in_cluster.append(feature_vectors[y])
            # dimensions = feature_vectors['002492.txt'].size
            dimensions = 0
            for key, value in feature_vectors.items():
                dimensions = value.size
                break
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
            distance = self.compute_euclidean_distance(centroid, centroid_temp)
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