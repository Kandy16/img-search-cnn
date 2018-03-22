import os
import config



# --------------------Testing KNNN-----------------

from ml.knn import knn
obj = knn.KNN()
# print(obj.get_random_images(10))
obj.prepare_data_for_KNN(config.KNN_IMG_VECTORS_FILEPATH , config.KNN_DATASET_PATH  , "" , "fc8")

# relevant_images = ["test_1742.txt"]
# print(obj.get_feedback(relevant_images , config.KNN_DATASET_PATH))



# -------------TESTING COSINE---------------

# Lets test cosine similarity which first needs to create the nearest neighbour for each image vector. This is a pre process.

#from ml.cosine import cosine_similarity_cluster

#obj_cosine = cosine_similarity_cluster.CosineSimilarityCluster()
#obj_cosine.nearest_neighbours_for_each_imagevector(config.COSINE_IMG_VECTORS_FILEPATH , config.COSINE_NEAREST_NEIGHBOUR_SAVE_PATH , model = "" , layer = "fc8")
#print(obj_cosine.get_feedback("/var/www/clone-img-search-cnn/img-search-cnn/webapp/dataset/cosine/cosine_nearest_neighbors/" , ["014999.jpg"]))


#-----------------Testing DATABASE ------------------

#from database.database import Database

#obj = Database()
#obj.fillin_database()