import sys
import os.path
import pdb
from .. import config
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_extraction')) # BUT WHYYYYYYYYYYYYYYY

#obj_fe.extract_features(EnumModels.Models.bvlc_alexnet.name , "fc8")
#obj_fe.extract_features(EnumModels.Models.ResNet18_ImageNet_CNTK_model.name , "z")  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
#obj_fe.extract_features(EnumModels.Models.bvlc_googlenet.name)
#obj_fe.extract_features(EnumModels.Models.bvlc_reference_caffenet.name)
#etobj_fe.extract_features(EnumModels.Models.finetune_flickr_style.name)

from ..features_extraction import EnumModels
extract_info = {"model_name":EnumModels.Models.finetune_flickr_style.name , "model_layer":"fc8_flickr"}

# Here is the code that will run for feature extraction and also for KNN vecotrs.p preparation and cosine similarity and clustering data preparation for a given model and layer
from ..features_extraction import feature_extraction
obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH , config.TEST_CAFEE_IMAGES_PATH , config.BASE_DIR) # Test config to real images file
obj_fe.extract_features(extract_info["model_name"] , extract_info["model_layer"])

# Prepare data for cosine similarity for given feature vectors as per model and layer provided
# import for cosine similarity
from ..ml.cosine import cosine_similarity_cluster as cs 
obj_cosine = cs.CosineSimilarityCluster(20) # object of KNN used for extract 
obj_cosine.nearest_neighbours_for_each_imagevector(config.COSINE_IMG_VECTORS_FILEPATH , config.COSINE_NEAREST_NEIGHBOUR_SAVE_PATH , extract_info["model_name"] , extract_info["model_layer"])

# Prepare data for KNN. vectors.p for given model and layer
# import for knn machine learning implementation
from ..ml.knn import knn
obj_knn = knn.KNN() # object of KNN used for search(random images), extract , feedback
obj_knn.prepare_data_for_KNN(config.KNN_IMG_VECTORS_FILEPATH , config.KNN_DATA_SAVE_PATH  , extract_info["model_name"] , extract_info["model_layer"])


# BEGIN    CLUSTERING EXAMPLEEEEEEEEEEEEEEEE
from ..ml.clustering import image_clustering as ic
obj_cluster = ic.ImageClustering(10)
obj_cluster.get_prepare_clusters(config.CLUSTERING_VECTORS_P_PATH , config.CLUSTERING_DATA_SAVE_PATH , extract_info["model_name"] , extract_info["model_layer"].replace("/" , "-"))

# END    CLUSTERING EXAMPLEEEEEEEEEEEEEEEE
		