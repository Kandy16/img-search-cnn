# TODO
# 0. able to select alorithm (KNN, Cosine similarity) defulat KNN
# 1. Extract feature (unique class jasma yo method huncha)
# 2. Extract method ma model type ()


import json
from datetime import datetime
import os

from PIL import Image
from flask import Response, abort
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

import config
from models.models import db
from models.models import Feedback, Image

# import for knn machine learning implementation
from ml.knn import knn

# import for cosine similarity
from ml.cosine import cosine_similarity_cluster as cs

app_settings = {
    'algorithms': ['KNN', 'Cosine Similarity'],
    'current': 'KNN',
    'message': ''
}

# handling base directory i.e. location of the webapp folder in our case
basedir = config.BASE_DIR

# handling feedback_dir
feedback_dir = config.FEEDBACK_DIR

parent_path = "/".join(basedir.split('/')[:-1])  #IS it used? //TODO

# handling image directory
img_dir = config.CAFEE_IMAGES_PATH

# object of knnclassifier used for search and feedback
obj_knn = knn.KNN()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'weareLearningDeepLearxingokjalsf2oue'

# Database for app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite3')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}


@app.route('/<path:filename>', methods=['get', ])
def image(filename):
    try:
        return send_from_directory(img_dir, filename)
    except:
        abort(404)


@app.route('/', methods=['get', ])
def index():
    # note this file_dir sent as argument was to just for testing. Also remove {{file_dir}} from index.html when removing this.
    return render_template('pages/index.html' , file_dir= "")


@app.route('/search', methods=['POST', ])
def search():
    search_query = request.form.get('search')

    # condition 1
    # when query is totally new    
    rand_images = obj_knn.get_random_images(10)  #

    # condition 2
    # when query is already in database

    return render_template('pages/result.html', query=search_query, images=rand_images)


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


    obj = cs.CosineSimilarityCluster()
    #obj.nearest_neighbours_for_each_imagevector()
    # using filenames from neighbours json file test
    rand_images = obj.get_feedback("/var/www/clone-img-search-cnn/img-search-cnn/webapp/dataset/cosine/cosine_nearest_neighbors/fc8/" , images)
    #rand_images = ['000001.jpg']
    
    search_query = "cat"

    return render_template('pages/result.html', query=search_query, images=rand_images , image_number = images[0])


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
    if request.method == 'POST':
        app_settings['current'] = 'Cosine'

    return render_template('pages/settings.html', app_settings=app_settings)


@app.route('/extract', methods=['post', ])
def extract():
    message = 'Sucessfully extracted model'
    return render_template('pages/settings.html', message=message)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('pages/404.html'), 404


if __name__ == '__main__':
    migrate = Migrate(app, db)
    manager = Manager(app)
    manager.add_command('db', MigrateCommand)
    manager.run()
