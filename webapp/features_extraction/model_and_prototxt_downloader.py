import sys
import os.path
import numpy as np
import zipfile
import shutil
from EnumModelsPrototxtURL import ModelsPrototxtURL
from EnumModels import Models
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

import pathlib2
import pdb

class ModelAndPrototxtDownloader(object):

    def __init__(self, download_path):
        self.download_path = download_path

    def download_model_and_prototxt(self, pretrained_model):
        def errhandler ():
            print("Invalid Model Name")

        takeaction = {
            ModelsPrototxtURL.bvlc_alexnet.name : self._download_for_alexnet,
            ModelsPrototxtURL.bvlc_googlenet.name :self._download_for_googlenet,
            ModelsPrototxtURL.bvlc_reference_caffenet.name :self._download_for_reference_caffenet,
            ModelsPrototxtURL.finetune_flickr_style.name :self._download_for_finetune_flickr_style
            # Can add more model and also function associated to it

        }
        takeaction.get(pretrained_model , errhandler)()

    def _download_for_alexnet(self):
        # CAFFE ALEXNET
        self.download_model("bvlc_alexnet.caffemodel", "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel") 
        self.download_deploy_protxt("bvlc_alexnet_deploy.prototxt" , "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt")

    def _download_for_googlenet(self):
        #CAFEE GOOGLENET
        self.download_model("bvlc_googlenet.caffemodel", "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel")
        self.download_deploy_protxt("bvlc_googlenet_deploy.prototxt" , "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt")


    def _download_for_reference_caffenet(self):
        #CAFEE CAFFENET
        self.download_model("bvlc_reference_caffenet.caffemodel","http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel")
        self.download_deploy_protxt("bvlc_reference_caffenet_deploy.prototxt" , "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt")


    def _download_for_finetune_flickr_style(self):
        #CAFEE FLICKRSTYLE
        self.download_model("finetune_flickr_style.caffemodel" , "http://dl.caffe.berkeleyvision.org/finetune_flickr_style.caffemodel")
        self.download_deploy_protxt("finetune_flickr_style_deploy.prototxt" , "https://raw.githubusercontent.com/BVLC/caffe/master/models/finetune_flickr_style/deploy.prototxt")




    # Note It will download to the same location as this python file
    def download_model(self, model_file_name, model_url):
        # model_dir = os.path.dirname(os.path.abspath(__file__))
        # filename = os.path.join(model_dir, model_file_name)
        
        filename = self.download_path
        pathlib2.Path(filename).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(filename , model_file_name)
        if not os.path.exists(filename):
            print('Downloading model from ' + model_url + ', may take a while...')
            pdb.set_trace()
            urlretrieve(model_url, filename)
            print('Saved model as ' + filename)
        else:
            print('\n' + model_file_name + ' model already available at ' + filename)


    # Note It will download to the same location as this python file
    def download_deploy_protxt(self, prototxt_file_name, prototxt_url):
        # model_dir = os.path.dirname(os.path.abspath(__file__))
        # filename = os.path.join(model_dir, prototxt_file_name)
        filename = self.download_path
        pathlib2.Path(filename).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(filename ,prototxt_file_name)

        if not os.path.exists(filename):
            print('Downloading prototxt from ' + prototxt_url + ', may take a while...')
            urlretrieve(prototxt_url, filename)
            print('Saved prototxt as ' + filename)
        else:
            print(prototxt_file_name + ' prototxt already available at ' + filename)




# Here we download two model from CNTK.  
#						1. AlexNet_ImageNet_CNTK.model (Have to have minibatch source detail as image_height = 227 and image width = 227)
#						2. ResNet18_ImageNet_CNTK.model  (Have to have minibatch source detail as image_height = 224 and image width = 224)



# CNTK RESNET18
#download_model("ResNet18_ImageNet_CNTK.model", "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model")

# CNTK ALEXNET
#download_model("AlexNet_ImageNet_CNTK.model", "https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model")





