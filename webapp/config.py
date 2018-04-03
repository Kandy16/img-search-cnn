import os
import pdb
BASE_DIR = os.path.dirname(__file__) # Till /webapp // When running fomr app.py

BASE_DIR = "/var/www/img-search-cnn/webapp/"  # can delete after all testing done

CAFEE_IMAGES_PATH = os.path.join(BASE_DIR, "dataset" ,"images") #os.path.join(BASE_DIR, 'dataset/images/')

FEEDBACK_DIR = os.path.join(BASE_DIR, "dataset" ,"feedbacks")
## -- BEGIN CONFIGURATION FOR FEATURE EXTRACTION AND DATA PREPARATION -- 

# For feature Extraction
FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH =  os.path.join(BASE_DIR, "features_extraction" ,"models")
TEST_CAFEE_IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "images_eval") # Please use CAFEE_IMAGES_PATH in deployement and delete this # USUAGE so far 

# For Cosine Data Preparation
COSINE_IMG_VECTORS_FILEPATH = os.path.join(BASE_DIR, "dataset" ,"features_etd1a") # Note after that we give model name and layer name so no need to mention here.
# now we need the location for where to save the cosine calculated nearest neighbour
COSINE_NEAREST_NEIGHBOUR_SAVE_PATH = os.path.join(BASE_DIR, "dataset" ,"COSINE")

# For KNN Data preparation
KNN_IMG_VECTORS_FILEPATH = COSINE_IMG_VECTORS_FILEPATH  # Same initial location
KNN_DATA_SAVE_PATH = os.path.join(BASE_DIR, "dataset" ,"KNN")

## -- END CONFIGURATION FOR FEATURE EXTRACTION AND DATA PREPARATION --

# For CLUSTERING Data Preparation
CLUSTERING_VECTORS_P_PATH = os.path.join(BASE_DIR, "dataset" ,"KNN")
CLUSTERING_DATA_SAVE_PATH = os.path.join(BASE_DIR, "dataset" ,"CLUSTERING")

# For TSNE DATA preparation
TSNE_COSINE_IMG_VECTORS_FILEPATH = os.path.join(BASE_DIR, "dataset" , "tSNE_visualization","features" , "reduced_features")
TSNE_COSINE_NEAREST_NEIGHBOUR_SAVE_PATH = os.path.join(BASE_DIR, "dataset" ,"tSNE_visualization" ,"COSINE")