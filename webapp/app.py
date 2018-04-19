# TODO
# 0. able to select alorithm (KNN, Cosine similarity) defulat KNN
# 1. Extract feature (unique class jasma yo method huncha)
# 2. Extract method ma model type ()

import pdb
import json
from datetime import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_extraction')) # BUT WHYYYYYYYYYYYYYYY ASKKKKKKKKKKKKKKKKKKKKKKK
from PIL import Image
from flask import Response, abort
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from flask import jsonify
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from utils.utils import split_array_equally

import config
from database.models.models import db
from database.models.models import Feedback, Image, Base, NeuralLayer, NeuralNetworkModel, DefaultSettings

from database.models.models import QueryString, FeatureVectorsQueryString, ApplicationVideo

from sqlalchemy import event, DDL
import caffe

app_settings = {
    'algorithms': ['KNN', 'Cosine Similarity'],
    'current': 'KNN',
    'message': ''
}

# handling base directory i.e. location of the webapp folder in our case
basedir = config.BASE_DIR

# handling feedback_dir
feedback_dir = config.FEEDBACK_DIR

# handling image directory Choose depending on the images used
#img_dir = config.CAFEE_IMAGES_PATH
img_dir = config.TEST_CAFEE_IMAGES_PATH

# TODO only for test
# img_dir = os.path.join(basedir, 'images/')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'weareLearningDeepLearxingokjalsf2oue'

# Database for app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite3')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


# Database fill up initially with default data.
from database.database import Database

# imports for feature extraction
from features_extraction import feature_extraction
from features_extraction import EnumModels

# import for knn machine learning implementation
from ml.knn import knn
obj_knn = knn.KNN() # object of KNN used for search(random images), extract , feedback

# import for cosine similarity
from ml.cosine import cosine_similarity_cluster as cs
obj_cosine = cs.CosineSimilarityCluster() # object of KNN used for extract , feedback

# Import for application - Youtube extraction
from application.images_youtube_extract import ImagesYoutubeExtract

# Import for random images variations
from ml.clustering import random_images
vectors_save_location = config.KNN_DATA_SAVE_PATH
clusters_save_location = config.CLUSTERING_DATA_SAVE_PATH
modelname = "bvlc_alexnet"
layername = "fc7"
obj_random_images = random_images.RandomImages(vectors_save_location , clusters_save_location , modelname , layername, number_of_clusters=10) # object of KNN used for extract , feedback


# default query object
query_object = {
    'model': 'some model',
    'layer': 'some layer',
    'algo': 'some algo'
}

@app.before_first_request
def setup():
    # Recreate database each time for demo..
    # TODO Remove this once table is created.
    Base.metadata.drop_all(bind=db.engine)
    Base.metadata.create_all(bind=db.engine)
    # Till Here Remove

    obj_database = Database()
    obj_database.fillin_database(db)


@app.context_processor
def inject_now():
    def capitalize(word):
        words = word.split('_')
        words = map(lambda w: w.capitalize(), words)
        return " ".join(words)

    available_models = dict()
    models_names = []
    all_neural_models = db.session.query(NeuralNetworkModel).all()

    for x in db.session.query(DefaultSettings).all():
        default_settings = {"model_name" : x.model_name , "layer_name": x.layer_name , "ml_algorithm" : x.ml_algorithm}

    for model in all_neural_models:
        models_names.append(model.name)
        available_models[model.name] = [{'name': x.name, 'extracted': x.extracted } for x in model.neural_network]
    return {'now': datetime.utcnow(), 'available_models': json.dumps(available_models), 'models_names': models_names, 'capitalize': capitalize , 'default_settings':default_settings}

@app.route('/<path:filename>', methods=['get', ])
def image(filename):
    try:
        return send_from_directory(img_dir, filename)
    except:
        abort(404)

@app.route('/suggestion', methods=['get', ])
def suggestion():
    term = request.args.get('term', '', type=str)
    # Call the database
    # Store in variable
    # result = ['hello', 'world']
    return jsonify(['hello', 'world', 'this'])

@app.route('/apps', methods=['get', ])
def apps():
    return jsonify([{'images': ['jpt', 'haha']}, {'related': ['hait', 'jait']}])

@app.route('/', methods=['get', ])
def index():
    # note this file_dir sent as argument was to just for testing. Also remove {{file_dir}} from index.html when removing this.
    return render_template('pages/index.html' , file_dir= "")


@app.route('/search', methods=['POST', ])
def search():
    search_query = request.form.get('search')
    # ml_settings = request.form.get('ml_settings')


    query_exists = db.session.query(db.exists().where(QueryString.query_string == search_query)).scalar()
    # condition 1
    # when query is totally new
    if not query_exists:
        obj_query_string = QueryString(query_string=search_query)
        db.session.add(obj_query_string)
        db.session.commit()
        rand_images = obj_random_images.get_n_random_images_full_random(config.TEST_CAFEE_IMAGES_PATH , 10)

    else:
    	# Now load the random images from existing query
    	rand_images = []
    	obj_query_string = db.session.query(QueryString).filter_by(query_string=search_query).first()
    	queried_images = db.session.query(FeatureVectorsQueryString).filter_by(feature_vectors_id=obj_query_string.id)
    	if queried_images.first() is None:
    		rand_images = obj_random_images.get_n_random_images_full_random(config.TEST_CAFEE_IMAGES_PATH , 10)
        else:
        	for obj in queried_images:
    			rand_img = obj.feature_vector_filename.split(".")[0] + '.jpg'
    			rand_images.append(rand_img)        	
    
    related_images = obj_random_images.get_n_random_images_full_random(config.TEST_CAFEE_IMAGES_PATH , 10)


    # condition 2
    # when query is already in database

    splitted_images = split_array_equally(rand_images, 3)
    return render_template('pages/result.html', query=search_query, images=splitted_images, related_images=related_images)


@app.route('/feedback', methods=['POST', ])
def feedback():

    # Code to test cosine similarity
    # First is to ask if the json file has been created usually with the following code.
    try:
        feedback_raw = request.form.to_dict()
        feedback_dict = json.loads(feedback_raw['feedback'])
    except:
        abort(404)


    #neighbour = for_feedback(feedback_dict['images'])
    query = feedback_dict['query']
    images = feedback_dict['images']
    related_images_feedback = feedback_dict['related_images']

    # get object row for the query
    obj_query_string = db.session.query(QueryString).filter_by(query_string=query).first()


    # using filenames from neighbours json file test
    calculated_cosine_neighbours_path = os.path.join(config.COSINE_NEAREST_NEIGHBOUR_SAVE_PATH , "bvlc_alexnet" , "fc7")
    rand_images = obj_cosine.get_feedback(calculated_cosine_neighbours_path , images)

    db.session.query(FeatureVectorsQueryString).filter_by(feature_vectors_id=obj_query_string.id).delete()
    db.session.commit()

    for imgs in rand_images:
        feature_vector = imgs.split(".")[0] + '.txt'
        obj_feature_vector_query_string = FeatureVectorsQueryString(feature_vector_filename=feature_vector , model_name = "bvlc_alexnet" , model_layer = "fc7" , feature_vectors = obj_query_string)
        db.session.add(obj_feature_vector_query_string)
        db.session.commit()
    
    related_images = obj_random_images.get_n_random_images_full_random(config.TEST_CAFEE_IMAGES_PATH , 10)
    splitted_images = split_array_equally(rand_images, 3)
    return render_template('pages/result.html', query=query, images=splitted_images, related_images=related_images , hello = images)


    # NEED TO ASK MADHU ABOUT DATABASE
    # feedback_raw = request.form.to_dict()
    # feedback_dict = json.loads(feedback_raw['feedback'])
    # neighbour = for_feedback(feedback_dict['images'])
    # query = feedback_dict['query']
    # images = feedback_dict['images']
    # query_vector = ''

    # feedback = db.session.query(Feedback).filter_by(query=query).first()

    # if feedback:
    #     feedback.query_vector = query_vector
    # else:
    #     feedback = Feedback(query=query, feature_vector=query_vector)

    # db.session.add(feedback)
    # db.session.commit()

    # db.session.add(feedback)
    # db.session.commit()
    # db_images = []
    # for img in images:
    #     db_images.append(Image(image_url=img, feedback_id=feedback.id))

    # db.session.add_all(db_images)
    # db.session.commit()

    # rand_images = ["014999.jpg" , "000090.jpg"]
    # for val in neighbour:
    #     rand_images.append(val[1].replace('txt', 'jpg'))
    # search_query = "cat"
    # return render_template('pages/result.html', query=search_query, images=rand_images)


@app.route('/settings', methods=['post', 'get'])
def settings():
    # Have to check if the folders exist and accordingly we have to update database of neural layer i.e. extracted status to true.
    allrow = "normal"
    if request.method == 'POST':
        try:
            default_settings_raw = request.form.to_dict()
            default_settings_dict = json.loads(default_settings_raw["default_settings"])
        except:
            abort(404)

        app_settings['current'] = 'Cosine'
        allrow = default_settings_dict

        # Need to update database with new data received from server as post i.e when default button is pressed
        target_row = db.session.query(DefaultSettings).all()[0]
        target_row.model_name = default_settings_dict["model"]
        target_row.layer_name = default_settings_dict["layer"]
        target_row.ml_algorithm = default_settings_dict["algo"]
        db.session.commit()
        #default_settings = {"model_name" : default_settings_dict["model"] , "layer_name": default_settings_dict["layer"] , "ml_algorithm" : default_settings_dict["algo"]}
    else:
        # Obtaining the default settings from the database
        #for x in db.session.query(DefaultSettings).all():
            #default_settings = {"model_name" : x.model_name , "layer_name": x.layer_name , "ml_algorithm" : x.ml_algorithm}

        load_all_rows = db.session.query(NeuralLayer).all()
        for row in load_all_rows:
            extract_from_layer = row.name
            pretrained_model = row.neural_network.name
            smoothed_layer_name = extract_from_layer.replace("/" , "-")
            filename = os.path.join(config.BASE_DIR, "dataset", "features_etd1a" ,  pretrained_model, smoothed_layer_name)
            if os.path.exists(filename):
                # This is where we start updating database extracted boolean in neural layer.
                target_row = db.session.query(NeuralLayer).filter_by(id = row.id).first()
                target_row.extracted = True
                db.session.commit()
            else:
                target_row = db.session.query(NeuralLayer).filter_by(id = row.id).first()
                target_row.extracted = False
                db.session.commit()
    return render_template('pages/settings.html', app_settings=app_settings , allrow = allrow )


@app.route('/extract', methods=['post', ])
def extract():
    try:
        feedback_raw = request.form.to_dict()
        feedback_dict = json.loads(feedback_raw["extract_settings"])
    except:
        abort(404)
    query = feedback_dict['model']
    images = feedback_dict['layer']

    # Here as a post we expect a dictionary
    #For alexnet
    #extract_info = {"model_name":EnumModels.Models.bvlc_alexnet.name , "model_layer":"fc8"}

    #For bvlc_googlenetextract_info = {"model_name":EnumModels.Models.bvlc_reference_caffenet.name , "model_layer":"fc8"}
    #
    # Random images in search - check if normal or clustered random images.

    # Step 1 First we need to extract features depending on the given model and layer - internally it downloads model and prototxt, creates images.txt.
    # Step 2 after extraction we need to prepare data for getting random images i.e. clustering
    #                                    prepare data for KNN i.e. vectors.p file.
    #                                    prepare data for cosine i.e. nearest neighbours for all image vectors.
    #

    #obj_fe = feature_extraction.FeatureExtraction(config.FEATURE_EXTRACTION_MODELS_DOWNLOAD_PATH , config.TEST_CAFEE_IMAGES_PATH , config.BASE_DIR) # Test config to real images file

    #obj_fe.extract_features(extract_info["model_name"] , extract_info["model_layer"])

    # Prepare data for cosine similarity for given feature vectors as per model and layer provided
    #obj_cosine.nearest_neighbours_for_each_imagevector(config.COSINE_IMG_VECTORS_FILEPATH , config.COSINE_NEAREST_NEIGHBOUR_SAVE_PATH , extract_info["model_name"] , extract_info["model_layer"])

    # Prepare data for KNN. vectors.p for given model and layer
    #obj_knn.prepare_data_for_KNN(config.KNN_IMG_VECTORS_FILEPATH , config.KNN_DATA_SAVE_PATH  , extract_info["model_name"] , extract_info["model_layer"])

    #message = 'Sucessfully extracted model' + extract_info["model_name"] + extract_info["model_layer"]

    message = query + images
    return render_template('pages/settings.html', app_settings=app_settings , message=message)


@app.route('/application', methods=['get',])
def application():
    images_save_location = "/var/www/img-search-cnn/webapp/dataset/applicationData"
    #youtube_url = "https://www.youtube.com/watch?v=kQcUamGg7Yw"
    obj_iye = ImagesYoutubeExtract(images_save_location)
    urls , origurls = obj_iye.get_urls_search_query("monkey") 
    obj_iye.extract_images_youtube(origurls[0] , "salgaris")
    splitted_urls = split_array_equally(urls, 3)


    #urls = [u'https://www.youtube.com/embed/axgFo7QazQo', u'https://www.youtube.com/embed/1Wh8RzcQZr4', u'https://www.youtube.com/embed/WEkSYw3o5is', u'https://www.youtube.com/embed/wZZ7oFKsKzY', u'https://www.youtube.com/embed/jpYDw7AJDtM', u'https://www.youtube.com/embed/z3U0udLH974', u'https://www.youtube.com/embed/g-6C9LaGIJU', u'https://www.youtube.com/embed/5dsGWM5XGdg', u'https://www.youtube.com/embed/5530I_pYjbo', u'https://www.youtube.com/embed/i-AXImNxCAE', u'https://www.youtube.com/embed/sHWEc-yxfb4', u'https://www.youtube.com/embed/3vDV1F_fngc', u'https://www.youtube.com/embed/XyNlqQId-nk', u'https://www.youtube.com/embed/O1KW3ZkLtuo', u'https://www.youtube.com/embed/EtH9Yllzjcc', u'https://www.youtube.com/embed/jFm3HDLph0M', u'https://www.youtube.com/embed/72NfSwCzFVE', u'https://www.youtube.com/embed/OqQPv78AMw0']
    return render_template('pages/application.html' , urls = splitted_urls)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('pages/404.html'), 404

if __name__ == '__main__':
    migrate = Migrate(app, db)
    manager = Manager(app)
    manager.add_command('db', MigrateCommand)
    manager.run()
    # app.run(debug = True)
