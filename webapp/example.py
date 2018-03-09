from feature_extraction import image_list_creator
from feature_extraction import feature_extraction
from feature_extraction import EnumModels
import config

# First we need to create a image list of all the images in a folder. 
# Creates image.txt listing the location of all images 
il = image_list_creator.ImageListCreator()
il.make_list_image_filenames(config.TEST_CAFEE_IMAGES_PATH)


# Now we have the images.txt file with all images
# We extract features from theses images on various models

obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH)
# Param 1 : Takes model name from EnumModels.
# Param 2 : expected that models is the folder that contains this model and prototxt file. 
#           But if it doesnot exist it will create a folder models and download necessary files


obj_fe.extract_features(EnumModels.Models.bvlc_alexnet.name)
obj_fe.extract_features(EnumModels.Models.bvlc_googlenet.name)
obj_fe.extract_features(EnumModels.Models.bvlc_reference_caffenet.name)
obj_fe.extract_features(EnumModels.Models.finetune_flickr_style.name)

# Can do other models as given in Enum Models class

