import os, numpy, operator, random, pickle
basedir = os.getcwd()

import config



class KNNClassifier:

    def __init__(self, path_to_features):
        self._path_to_features = path_to_features


    def get_random_images(self, feature_vectors=None):
        related_images = []
        if feature_vectors is None:
            return related_images
        else:
            return related_images


    # query word "cat"
    # array of images []
    # array of features vector []

    def get_related_images():
        return []




try:
    pathToFeatures = config.CAFEE_FC8_PATH
except Exception as e:
    # TODO change the directory to the good one
    pathToFeatures = os.path.join('../', basedir, 'images/')


# Helper Utilities

# returns an array of random images filename (of the format xxxxxx.jpg)
def display_random_images(start, end, number):
    images = []
    for x in range(0,number):
        irand = random.randrange(start, end)
        images.append(str(irand).zfill(6) + ".jpg")
    #print(images)
    return images

## for namespacing and very naive implementation . TODO refactor this piece of trash
def for_feedback(images_selected):
    cluster = {}
    n = 0
    for x in findClusterForQuery('whatever'):
        cluster[x] = buildNumpyArrayForFeaturesByFileName(x)
         # cluster.append((x, buildNumpyArrayForFeaturesByFileName(x)))
        n = n+1
         # print(n)


    k = 10

    irrelevant = []
    relevant = []

    #print("Image selected:")
    #print(images_selected)
    print("Cluster printing")
    #print(cluster.keys())

    for image in images_selected:
        if image.replace('jpg', 'txt') in cluster.keys():
            relevant.append(cluster[image.replace('jpg', 'txt')])

    dimensions = cluster['000001.txt'].size
    print("####################")
    #print(relevant)
    neighbours = findKNeighbours(k, cluster, buildQueryVector(relevant, irrelevant), 'euclidean')
    #print(neighbours)
    return neighbours



def buildNumpyArrayForFeaturesByFileName(filename):
    f = open(pathToFeatures + filename, 'r')
    data = f.read()
    f.close()
    return numpy.fromstring(data, dtype=float, sep='\n')

# this function calculates the euclidean distance between 2 feature vectors
def calculateEuclideanDistance(features1, features2):
    return numpy.linalg.norm(features1 - features2)

# this function will find the k closest neighbours for a given query vector in a certain cluster
def findKNeighbours(k, cluster, query_vector, distance_metric):
    # building initial array for 5 neighbours
    neighbours = []
    for x in range(0, k):
        neighbours.append((float('inf'), '0'))

    # calculating distance between initial query vector and all images within its cluster
    for image_name, feature_vector in cluster.items():
        # using different distance metrices
        distance = float('inf')
        if distance_metric == 'euclidean':
            distance = calculateEuclideanDistance(query_vector, feature_vector)
        else:
            distance = calculateEuclideanDistance(query_vector, feature_vector)

        if (distance < neighbours[k - 1][0]):
            neighbours[k - 1] = (distance, image_name)
            neighbours.sort(key=operator.itemgetter(0))
    return neighbours


def findClusterForQuery(query_vector):
    # TODO
    cluster = []
    x = 0
    for filename in sorted(os.listdir(pathToFeatures)):
        cluster.append(filename)
        if x == 10000:
            break
        x += 1
    return cluster


# this function builds a new query_vector based on the relevant images using the average
def buildQueryVector(relevantFeatures, irrelevantFeatures):
    #dimensions = cluster['000000.txt'].size
    dimensions = 1000
    query_vector = numpy.zeros(dimensions)
    if len(relevantFeatures) == 0:
        return 0
    for x in relevantFeatures:
        query_vector = numpy.add(query_vector, x)
    # for x in irrelevantFeatures:
    #     query_vector = numpy.subtract(query_vector, x)
    #n = len(relevantFeatures)
    #query_vector = numpy.divide(query_vector, n)
    return query_vector
