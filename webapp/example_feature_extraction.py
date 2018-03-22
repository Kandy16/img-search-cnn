from feature_extraction import feature_extraction
from feature_extraction import EnumModels
import config


# Now we have the images.txt file with all images
# We extract features from theses images on various models

obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH , config.TEST_CAFEE_IMAGES_PATH , config.BASE_DIR)
# Param 1 : Takes model name from EnumModels.
# Param 2 : expected that models is the folder that contains this model and prototxt file. 
#           But if it doesnot exist it will create a folder models and download necessary files


obj_fe.extract_features(EnumModels.Models.bvlc_alexnet.name , "fc8")
#obj_fe.extract_features(EnumModels.Models.bvlc_googlenet.name)
#obj_fe.extract_features(EnumModels.Models.bvlc_reference_caffenet.name)
#etobj_fe.extract_features(EnumModels.Models.finetune_flickr_style.name)

# Can do other models as given in Enum Models class

