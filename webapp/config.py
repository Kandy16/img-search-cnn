import os

BASE_DIR = os.path.dirname(__file__)

CAFEE_IMAGES_PATH = os.path.join(BASE_DIR, 'dataset/images/')

TEST_CAFEE_IMAGES_PATH = os.path.join(BASE_DIR, 'dataset/images-tryF/')  # can delete once everything is working. should use images at CAFEE_IMAGES_PATH just above this.

FEEDBACK_DIR = os.path.join(BASE_DIR , 'feedbacks/')

FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH =  os.path.join(BASE_DIR, 'feature_extraction/')

KNN_FEATURE_PATH = os.path.join(BASE_DIR, 'dataset/imagevectors/')

KNN_DATASET_PATH = os.path.join(BASE_DIR, 'dataset/KNN/')