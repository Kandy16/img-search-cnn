import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), 'features_extraction')) # BUT WHYYYYYYYYYYYYYYY
from features_extraction import feature_extraction
from features_extraction import EnumModels
import config

# import for knn machine learning implementation
from ml.knn import knn
obj_knn = knn.KNN() # object of KNN used for search(random images), extract , feedback

# import for cosine similarity
from ml.cosine import cosine_similarity_cluster as cs 
obj_cosine = cs.CosineSimilarityCluster() # object of KNN used for extract 

# Now we have the images.txt file with all images
# We extract features from theses images on various models

#obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH , config.TEST_CAFEE_IMAGES_PATH , config.BASE_DIR)
# Param 1 : Takes model name from EnumModels.
# Param 2 : expected that models is the folder that contains this model and prototxt file. 
#           But if it doesnot exist it will create a folder models and download necessary files


#obj_fe.extract_features(EnumModels.Models.bvlc_alexnet.name , "fc8")
#obj_fe.extract_features(EnumModels.Models.ResNet18_ImageNet_CNTK_model.name , "z")  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
#obj_fe.extract_features(EnumModels.Models.bvlc_googlenet.name)
#obj_fe.extract_features(EnumModels.Models.bvlc_reference_caffenet.name)
#etobj_fe.extract_features(EnumModels.Models.finetune_flickr_style.name)

# Can do other models as given in Enum Models class



extract_info = {"model_name":EnumModels.Models.bvlc_reference_caffenet.name , "model_layer":"fc8"}
# Here is the code that will run for feature extraction and also for KNN vecotrs.p preparation and cosine similarity data preparation for a given model and layer
obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH , config.TEST_CAFEE_IMAGES_PATH , config.BASE_DIR) # Test config to real images file

obj_fe.extract_features(extract_info["model_name"] , extract_info["model_layer"])

# Prepare data for cosine similarity for given feature vectors as per model and layer provided
obj_cosine.nearest_neighbours_for_each_imagevector(config.COSINE_IMG_VECTORS_FILEPATH , config.COSINE_NEAREST_NEIGHBOUR_SAVE_PATH , extract_info["model_name"] , extract_info["model_layer"])

# Prepare data for KNN. vectors.p for given model and layer
obj_knn.prepare_data_for_KNN(config.KNN_IMG_VECTORS_FILEPATH , config.KNN_DATA_SAVE_PATH  , extract_info["model_name"] , extract_info["model_layer"])



