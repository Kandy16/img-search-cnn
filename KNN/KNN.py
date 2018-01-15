import os, numpy, operator, random, pickle
from PIL import Image
from datetime import datetime

# this is the path where you have stored the features
pathToFeatures = 'features/fc8/'
# images need to be stored in /media/images/



# this function will look up in which cluster a given query vector or feature vector is
def findClusterForQuery(query_vector):
    # TODO
    cluster = []
    for filename in os.listdir(os.getcwd() + '/' + pathToFeatures):
        cluster.append(filename)
    return cluster


# this function builds a numpy array for a certain feature vector
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


# this function selects k random images
def findRandomImages(k, cluster):
    random_images = []
    for n in range(0, k):
        filename, feature_vector = random.choice(list(cluster.items()))
        random_images.append((0, filename))
    return random_images


# this function builds a new query_vector based on the relevant images using the average
def buildQueryVector(relevantFeatures, irrelevantFeatures):
    dimensions = cluster['000000.txt'].size
    query_vector = numpy.zeros(dimensions)
    if len(relevantFeatures) == 0:
        return 0
    for x in relevantFeatures:
        query_vector = numpy.add(query_vector, x)
    # for x in irrelevantFeatures:
    #     query_vector = numpy.subtract(query_vector, x)
    n = len(relevantFeatures)
    query_vector = numpy.divide(query_vector, n)
    return query_vector


# script

cluster = {}
# this will build the numpy array for every feature vector and store it in a pickle file, you need to activate this on your first run
# n = 0
# for x in findClusterForQuery('whatever'):
#     cluster[x] = buildNumpyArrayForFeaturesByFileName(x)
#     # cluster.append((x, buildNumpyArrayForFeaturesByFileName(x)))
#     n = n+1
#     # print(n)

# pickle.dump(cluster, open("vectors.p", "wb"))

cluster = pickle.load(open("vectors.p", "rb"))
query_word = input("Query Word: ")

k = 10

relevant = []
irrelevant = []


neighbours = findRandomImages(k, cluster)

while True:
    print('THE NEXT ITERATION OF IMAGES BEGINS: ')
    relevant = []
    irrelevant = []
    for distance, filename in neighbours:
        img = Image.open('media/images/' + filename[:-3] + 'jpg')
        img.show()
        feedback = input('Is the image relevant to your query? Irrelevant = 0, Relevant = 1, Unsure = 2')
        if feedback == '0':
            feature_vector = cluster[filename]
            irrelevant.append(feature_vector)
        elif feedback == '1':
            feature_vector = cluster[filename]
            relevant.append(feature_vector)
    neighbours = findKNeighbours(k, cluster, buildQueryVector(relevant, irrelevant), 'euclidean')