import os
import json
from PIL import Image
from io import StringIO
from flask import Response, abort
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory

from random import randrange


app = Flask(__name__)
app.config['SECRET_KEY'] = 'weareLearningDeepLearxingokjalsf2oue'

## handling static images
current_directory = os.getcwd()

## handleing feedback_dir
feedback_dir = current_directory + '/feedbacks/'

parent_path = "/".join(current_directory.split('/')[:-1])
#file_dir = parent_path + '../../wrok/caffee/dataset/' + 'images/'
file_dir = "/home/codingyoddha/wrok/caffee/dataset/images/"
@app.route('/<path:filename>', methods=['get', ])
def image(filename):
    try:
        return send_from_directory(file_dir, filename)
    except:
        abort(404)


@app.route('/', methods=['get', ])
def index():
    return render_template('pages/index.html')

@app.route('/search', methods=['POST', ])
def search():
    search_query = request.form.get('search')
    rand_images = ['000001.jpg']
    rand_images = display_random_images(0,1000,10)
    return render_template('pages/result.html', query=search_query, images=rand_images)

@app.route('/feedback', methods=['POST', ])
def feedback():
    feedback_raw = request.form.to_dict()
    feedback_dict = json.loads(feedback_raw['feedback'])
    print(feedback_dict)
    neighbour = for_feedback(feedback_dict['images'])
    query = feedback_dict['query']
    images = feedback_dict['images']
    #if query is not None:
    #    return "There is no query "
    #return "thankyou for your feedback"

    rand_images = []
    for val in neighbour:
        rand_images.append(val[1].replace('txt', 'jpg'))
    #print(rand_images)
    search_query = "cat"
    return render_template('pages/result.html', query=search_query, images=rand_images)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('pages/404.html'), 404

# Helper Utilities

# returns an array of random images filename (of the format xxxxxx.jpg)
def display_random_images(start, end, number):
    images = []
    for x in range(0,number):
        irand = randrange(start, end)
        images.append(str(irand).zfill(6) + ".jpg")
    #print(images)
    return images


# adapated from Florian's work




import os, numpy, operator, random, pickle



pathToFeatures = '/home/codingyoddha/wrok/caffee/features/fc8/'


## for namespacing and very naive implementation . TODO Fuckign refactor this piece of trash
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


if __name__ == '__main__':
    # TODO: Please remove while going to production
    app.run(debug=True)
