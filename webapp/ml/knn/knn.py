import os
import random
import numpy
import operator
import pathlib2
# import config
import pickle

class KNN(object):

    def __init__(self):
        pass

    # this function builds a numpy array for a certain feature vector
    def _buildNumpyArrayForFeaturesByFileName(self, filename_path):
        f = open(filename_path, 'r')
        data = f.read()
        f.close()
        return numpy.fromstring(data, dtype=float, sep='\n')


    def prepare_data_for_KNN(self , image_vector_path , save_location , modelname , layername):
        # TODO  check if the feature_ oath is set to not NONE
        # TODO # Do we build up the vectors.p for all our models  here?

        vectors = {}
        # this will build the numpy array for every feature vector and store it in a pickle file, you need to activate this on your first run
        filename = os.path.join(save_location ,modelname,layername, "vectors.p")
        image_vector_path = os.path.join(image_vector_path , modelname , layername)
        if not os.path.exists(filename):
            m=0;
            for filename_vector in os.listdir(image_vector_path):
                m = m + 1
                print(m)
                vectors[filename_vector] = self._buildNumpyArrayForFeaturesByFileName(os.path.join(image_vector_path, filename_vector))	

            pathlib2.Path(os.path.join(save_location, modelname , layername)).mkdir(parents=True, exist_ok=True)	
            pickle.dump(vectors, open(filename, "wb"))
            print('Saved vectors.p at ' + filename)
        else:
            print('\n' + "vectors.p" + ' already available at ' + filename)


    def get_feedback(self, relevant_images , dumped_vector_path):
        #TODO list of relevant images. First we get all the nearest images for each relevant images and then do the intersection

        # For now only the first image from relevant images is taken into account.
        filename_rel = relevant_images[0]
        dict_feature_vectors = pickle.load(open(dumped_vector_path + "vectors.p", "rb"))

        feedback_first_feature_vec = dict_feature_vectors[filename_rel]
        neighbours = self._find_k_neighbours(4, dict_feature_vectors, feedback_first_feature_vec ,'euclidean') 

        return [os.path.splitext(neigbour)[0] + ".jpg" for (_, neigbour) in neighbours]

    # this function calculates the euclidean distance between 2 feature vectors
    def _calculate_euclidean_distance(self, features1, features2):
        return numpy.linalg.norm(features1 - features2)


    # this function will find the k closest neighbours for a given query vector in a certain cluster
    def _find_k_neighbours(self, k, cluster, query_vector, distance_metric):
        # building initial array for 5 neighbours
        neighbours = []
        for x in range(0, k):
            neighbours.append((float('inf'), '0'))

        # calculating distance between initial query vector and all images within its cluster
        for image_name, feature_vector in cluster.items():
            # using different distance metrices
            distance = float('inf')
            if distance_metric == 'euclidean':
                distance = self._calculate_euclidean_distance(query_vector, feature_vector)
            else:
                # Maybe some other distance measure.
                # TODO
                distance = self._calculate_euclidean_distance(query_vector, feature_vector)

            if (distance < neighbours[k - 1][0]):
                neighbours[k - 1] = (distance, image_name)
                neighbours.sort(key=operator.itemgetter(0))
        return neighbours


    @property
    def number_of_images(self):
        return self._number_of_images

    @number_of_images.setter
    def number_of_images(self, number_of_images):
        self._number_of_images = number_of_images
