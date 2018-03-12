#!/usr/bin/python
import sys
import logging

activate_this = '/var/www/clone-img-search-cnn/img-search-cnn/bin/activate_this.py'

with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0,"/var/www//clone-img-search-cnn/img-search-cnn/webapp/")

from app import app as application
# application.secret_key = 'haittsthisdeeplearningispsisingus'
