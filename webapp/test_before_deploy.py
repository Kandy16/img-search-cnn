import os
import config
from application.images_youtube_extract import ImagesYoutubeExtract



# --------------------Testing KNNN-----------------

#from ml.knn import knn
#obj = knn.KNN()
# print(obj.get_random_images(10))
#obj.prepare_data_for_KNN(config.KNN_IMG_VECTORS_FILEPATH , config.KNN_DATASET_PATH  , "" , "fc8")

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



# ------------------Testing Database update and delete
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models.models import QueryString, FeatureVectorsQueryString, ApplicationVideo, Base
 
engine = create_engine('sqlite:///data.sqlite3')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


not_processed_queries = [x.query_string for x in session.query(QueryString).filter_by(application_data_collected=False).all()]
print(not_processed_queries)

# Now we will use eash of these query words and update database with youtube urls

images_save_location = "/var/www/img-search-cnn/webapp/dataset/applicationData"
obj_iye = ImagesYoutubeExtract(images_save_location)
MAX_NUMBER_OF_URLS = 4
if not_processed_queries:	
	for query in not_processed_queries:
		embed_urls , origurls = obj_iye.get_urls_search_query(query , MAX_NUMBER_OF_URLS)
		print (len(origurls))

		#get object for the query in database
		obj_query_string = session.query(QueryString).filter_by(query_string=query).first()
		obj_query_string.application_data_collected = True
		session.commit()
		for idx, url in enumerate(origurls):
			obj_application_data = ApplicationVideo(youtube_url=url , youtube_embed_url = embed_urls[idx] ,application_videos = obj_query_string)
			session.add(obj_application_data)
			session.commit()
			obj_iye.extract_images_youtube(url , query) ## ORIGINAL
else:
	print("Everything up to date..")
	