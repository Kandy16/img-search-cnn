import sys
import os.path
import pdb
from .. import config
from ..features_extraction import EnumModels
from ..features_extraction import feature_extraction
from ..ml.cosine import cosine_similarity_cluster as cs 
from ..ml.knn import knn
from ..ml.clustering import image_clustering as ic

sys.path.append(os.path.join(os.path.dirname(__file__), 'features_extraction')) # BUT WHYYYYYYYYYYYYYYY

class DataPrepation(object):
	def __init__(self):
		pass

	def prepare_data(self, model_name , model_layer , layer_dimension):		
		extract_info = {"model_name":model_name , "model_layer":model_layer}

		# Here is the code that will run for feature extraction and also for KNN vecotrs.p preparation and cosine similarity and clustering data preparation for a given model and layer
		obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH , config.TEST_CAFEE_IMAGES_PATH , config.BASE_DIR) # Test config to real images file
		obj_fe.extract_features(extract_info["model_name"] , extract_info["model_layer"])

		# Prepare data for cosine similarity for given feature vectors as per model and layer provided
		obj_cosine = cs.CosineSimilarityCluster(layer_dimension) # object of KNN used for extract 
		obj_cosine.nearest_neighbours_for_each_imagevector(config.COSINE_IMG_VECTORS_FILEPATH , config.COSINE_NEAREST_NEIGHBOUR_SAVE_PATH , extract_info["model_name"] , extract_info["model_layer"].replace("/" , "-"))

		# Prepare data for KNN. vectors.p for given model and layer
		obj_knn = knn.KNN() # object of KNN used for search(random images), extract , feedback
		obj_knn.prepare_data_for_KNN(config.KNN_IMG_VECTORS_FILEPATH , config.KNN_DATA_SAVE_PATH  , extract_info["model_name"] , extract_info["model_layer"].replace("/" , "-"))


		# BEGIN    CLUSTERING EXAMPLEEEEEEEEEEEEEEEE
		obj_cluster = ic.ImageClustering(10)
		obj_cluster.get_prepare_clusters(config.CLUSTERING_VECTORS_P_PATH , config.CLUSTERING_DATA_SAVE_PATH , extract_info["model_name"] , extract_info["model_layer"].replace("/" , "-"))
		# END    CLUSTERING EXAMPLEEEEEEEEEEEEEEEE


if __name__ == "__main__":
	print("Option 1 : bvlc_alexnet -->  fc8")
	print("Option 2 : bvlc_alexnet -->  fc7")

	print("Option 3 : finetune_flickr_style -->  fc8_flickr")
	print("Option 4 : finetune_flickr_style -->  fc7")

	print("Option 5 : bvlc_reference_caffenet -->  fc8")
	print("Option 6 : bvlc_reference_caffenet -->  fc7")
	
	print("Option 7 : bvlc_googlenet -->  pool5/7x7_s1")

	print("Option 8 : ResNet18_ImageNet_CNTK_model -->  z")

	print("Option 9 : ALL IN ONE GO")

	obj_dp = DataPrepation()
	option_num = raw_input("Please choose from the above mentioned options : ")
	if option_num == "1" : 
		print("You choose option 1") 
		obj_dp.prepare_data("bvlc_alexnet" , "fc8" , 1000)
	elif option_num == "2" : 
		print("You choose option 2")
		obj_dp.prepare_data("bvlc_alexnet" , "fc7" , 4096)
	elif option_num == "3" : 
		print("You choose option 3")
		obj_dp.prepare_data("finetune_flickr_style" , "fc8_flickr" , 20)
	elif option_num == "4" : 
		print("You choose option 4")
		obj_dp.prepare_data("finetune_flickr_style" , "fc7" , 4096)
	elif option_num == "5" : 
		print("You choose option 5")
		obj_dp.prepare_data("bvlc_reference_caffenet" , "fc8" , 1000)
	elif option_num == "6" : 
		print("You choose option 6")
		obj_dp.prepare_data("bvlc_reference_caffenet" , "fc7" , 4096)
	elif option_num == "7" : 
		print("You choose option 7")
		obj_dp.prepare_data("bvlc_googlenet" , "pool5/7x7_s1" , 1024)
	elif option_num == "8" : 
		print("You choose option 8")
		obj_dp.prepare_data("ResNet18_ImageNet_CNTK_model" , "z" , 1000)
	elif option_num == "9" :
		print("You choose option 9")
		obj_dp.prepare_data("bvlc_alexnet" , "fc8" , 1000)
		obj_dp.prepare_data("bvlc_alexnet" , "fc7" , 4096)
		obj_dp.prepare_data("finetune_flickr_style" , "fc8_flickr" , 20)

		obj_dp.prepare_data("bvlc_reference_caffenet" , "fc8" , 1000)
		obj_dp.prepare_data("bvlc_reference_caffenet" , "fc7" , 4096)

		obj_dp.prepare_data("ResNet18_ImageNet_CNTK_model" , "z" , 1000)
 	else : print("None was chosen")