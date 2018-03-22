#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/var/www//caffe/python")
sys.path.insert(0,"/var/www//img-search-cnn/webapp/")

from app import app as application
# application.secret_key = 'haittsthisdeeplearningispsisingus'
