import os,sys
import numpy as np
import cntk as C
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs
import shutil    
import csv
import pathlib2
import glob

class CNTKFeatureExtraction(object):
    def __init__(self):
        pass


    def _download_model(self, model_file_name, model_url , model_download_path):
        try:
            from urllib.request import urlretrieve 
        except ImportError: 
            from urllib import urlretrieve

        pathlib2.Path(model_download_path).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(model_download_path, model_file_name)

        if not os.path.exists(filename):            
            print('Downloading model from ' + model_url + ', may take a while...')
            urlretrieve(model_url, filename)
            print('Saved model as ' + model_file_name)
        else:
            print('CNTK model already available at ' + filename)
        return model_file_name

    def create_mb_source(self, image_height, image_width, num_channels, map_file):
        transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
        return C.io.MinibatchSource(
            C.io.ImageDeserializer(map_file, C.io.StreamDefs(
                features=C.io.StreamDef(field='image', transforms=transforms),
                labels=C.io.StreamDef(field='label', shape=4096))),
            randomize=False)

    def eval_and_write(self, model_file, node_name, minibatch_source , map_file , features_folder_path):
        # load model and pick desired node as output
        loaded_model  = load_model(model_file)
        node_in_graph = loaded_model.find_by_name(node_name)
        output_nodes  = combine([node_in_graph.owner])

        # evaluate model and get desired node output
        print("Evaluating model for output node %s" % node_name)
        features_si = minibatch_source['features']

        with open(map_file) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)

        pathlib2.Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        for val in d:
            output_file = os.path.splitext(val[1])[0] + '.txt'
            output_file = os.path.join(features_folder_path , os.path.basename(output_file))

            with open(output_file, 'wb') as results_file:
                mb = minibatch_source.next_minibatch(1)
                output = output_nodes.eval(mb[features_si])
                # write results to file
                out_values = output[0].flatten()
                np.savetxt(results_file, out_values, newline = "\n")

    def _create_map_file(self, image_path , images_txt_save_path):
        filename = image_path

        if os.path.exists(filename):
            os.chdir(filename)
            text_file = open(os.path.join(images_txt_save_path , "cnn_images.txt"), "w")

            types = ('*.jpeg', '*.jpg' , '*.JPEG' , '*.png') # the tuple of file types
            files_grabbed = []
            for files in types:
                files_grabbed.extend(glob.glob(files))

            for file in files_grabbed:
                text_file.write("0\t" + os.path.join(filename , file) + "\t0" + "\n")
            text_file.close()
            print("images.txt created at location : " + images_txt_save_path)
            os.chdir(images_txt_save_path)
            return os.path.join(images_txt_save_path , "cnn_images.txt")
        else:
            print("No Folder exist of name \"" +  image_path + "\" Please create and put images into it")
            exit(0)

    def extract_feature(self, model_download_path , images_path, main_dir, pretrained_model , extract_from_layer):        
        model_file = self._download_model("ResNet18_ImageNet_CNTK.model", "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model" , model_download_path)
        
        # define location of model and data and check existence
        model_file  = os.path.join(model_download_path , model_file) 
        images_txt_save_path = os.path.join(main_dir , "features_extraction")  # TODO Really bad idea.. What if folder change?
        map_file = self._create_map_file(images_path, images_txt_save_path)
        if not (os.path.exists(model_file)):
            print("Cannot locate model")
            
        if not (os.path.exists(map_file)):
            print("Cannot locate mapped file for images")          

        # create minibatch source
        image_height = 224
        image_width  = 224
        num_channels = 3
        minibatch_source = self.create_mb_source(image_height, image_width, num_channels, map_file)

        # use this to print all node names of the model (and knowledge of the model to pick the correct one)
        node_outputs = get_node_outputs(load_model(model_file))
        for out in node_outputs: print("{0} {1}".format(out.name, out.shape))

     
        # evaluate model and write out the desired layer output
        features_folder_path = os.path.join(main_dir , "dataset", "features_etd1a" , pretrained_model , extract_from_layer)
        self.eval_and_write(model_file, extract_from_layer, minibatch_source , map_file , features_folder_path)

if __name__ == "__main__":
    main_dir = "C:\\Users\\Sabs\\Desktop\\img-search-cnn\webapp\\"
    obj = CNTKFeatureExtraction()
    
    obj.extract_feature( "C:\\Users\\Sabs\\Desktop\\img-search-cnn\webapp\\feature_extraction\\models\\", "D:\\Deep Learning DATA\\DLImages\\images_TRY\\" ,main_dir , "" , "z" )

