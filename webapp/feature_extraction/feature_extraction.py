# Imports
import sys
import pickle
import caffe
import numpy as np
import os
import pathlib2 # This just puts the feature vector in the form of array into the file.
from model_and_prototxt_downloader import ModelAndPrototxtDownloader
from EnumModels import Models

np.set_printoptions(threshold='nan')


class FeatureExtraction(object):
    def __init__(self , model_download_path):
        self.model_download_path = model_download_path
        self.mdl_proto_downloader = ModelAndPrototxtDownloader(model_download_path)
        # a little hack as we need model path with models folder.
        self.model_download_path = os.path.join(self.model_download_path , "models")


    def extract_features(self, pretrained_model):
        def errhandler (pretrained_model , _):
            print("Invalid Model Name - " + pretrained_model)

        takeaction = {
            Models.bvlc_alexnet.name : self._features_alexnet,
            Models.bvlc_googlenet.name : self._features_googlenet,
            Models.bvlc_reference_caffenet.name : self._features_reference_caffenet,
            Models.finetune_flickr_style.name : self._features_finetune_flickr_style
            # Can add more model and also function associated to it
        }
        takeaction.get(pretrained_model , errhandler)(pretrained_model)


    def _features_alexnet(self , pretrained_model):
        print("\n ----------Checking if appropriate pre trained model is downloaded and also the associated prototxt file --------")
        self.mdl_proto_downloader.download_model_and_prototxt(pretrained_model)

        extract_from_layer = "fc8"
        input_exp_file = "images.txt"
        model_def = os.path.join(self.model_download_path , "bvlc_alexnet_deploy.prototxt")
        pretrained_model = os.path.join(self.model_download_path , "bvlc_alexnet.caffemodel")
        batch_size = 10
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)

        # And now fc7
        extract_from_layer = "fc7"
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)


    def _features_googlenet(self , pretrained_model):
        print("\n ----------Checking if appropriate pre trained model is downloaded and also the associated prototxt file --------")
        self.mdl_proto_downloader.download_model_and_prototxt(pretrained_model)

        extract_from_layer = "pool5/7x7_s1"
        input_exp_file = "images.txt"
        model_def = os.path.join(self.model_download_path , "bvlc_googlenet_deploy.prototxt")
        pretrained_model = os.path.join(self.model_download_path , "bvlc_googlenet.caffemodel")
        batch_size = 10
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)


    def _features_reference_caffenet(self , pretrained_model):
        print("\n ----------Checking if appropriate pre trained model is downloaded and also the associated prototxt file --------")
        self.mdl_proto_downloader.download_model_and_prototxt(pretrained_model)

        extract_from_layer = "fc8"
        input_exp_file = "images.txt"
        model_def = os.path.join(self.model_download_path , "bvlc_reference_caffenet_deploy.prototxt")
        pretrained_model = os.path.join(self.model_download_path , "bvlc_reference_caffenet.caffemodel")
        batch_size = 10
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)

        # And now fc7
        extract_from_layer = "fc7"
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)


    def _features_finetune_flickr_style(self , pretrained_model):
        print("\n ----------Checking if appropriate pre trained model is downloaded and also the associated prototxt file --------")
        self.mdl_proto_downloader.download_model_and_prototxt(pretrained_model)

        extract_from_layer = "fc8_flickr"
        input_exp_file = "images.txt"
        model_def = os.path.join(self.model_download_path , "finetune_flickr_style_deploy.prototxt")
        pretrained_model = os.path.join(self.model_download_path , "finetune_flickr_style.caffemodel")
        batch_size = 10
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)

        # And now fc7
        extract_from_layer = "fc7"
        self._extract_features(pretrained_model, model_def , extract_from_layer , input_exp_file , batch_size)



    # global feature extraction method called by all models
    def _extract_features(self, pretrained_model, model_def, extract_from_layer, input_exp_file , batch_size ):
        
        # returns batch of image of size "batch_size"
        def _get_this_batch(image_list, batch_index, batch_size):
            start_index = batch_index * batch_size
            next_batch_size = batch_size    
            image_list_size = len(image_list)
            if(start_index + batch_size > image_list_size):  
                reamaining_size_at_last_index = image_list_size - start_index
                next_batch_size = reamaining_size_at_last_index
            batch_index_indices = range(start_index, start_index+next_batch_size,1)
            return image_list[batch_index_indices]

        # output file where we want to write extracted features
        output_pkl_file_name = "out.pkl"
        
        #print(os.cwd())
        ext_file = open(input_exp_file, 'r')
        image_paths_list = [line.strip() for line in ext_file]    
        ext_file.close()

        # required only if working in gpu mode .. default is cpu mode.
        #gpu_id = 4
        #caffe.set_mode_gpu();
        #caffe.set_device(gpu_id); 
        
        images_loaded_by_caffe = [caffe.io.load_image(im) for im in image_paths_list] 

        # create a net object 
        net = caffe.Net(model_def, pretrained_model, caffe.TEST)
        
        # Set up transformer - creates transformer object
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        # transpose image from HxWxC to CxHxW
        transformer.set_transpose('data', (2,0,1))
        # Swap image channels from RGB to BGR
        transformer.set_channel_swap('data', (2,0,1))
        # Set raw_scale = 255 to multiply with the values loaded with caffe.io.load_image
        transformer.set_raw_scale('data', 255)
        
        
        
        total_batch_nums = len(images_loaded_by_caffe)/batch_size    
        features_all_images = []
        images_loaded_by_caffe = np.array(images_loaded_by_caffe)
        # loop through all the batches 
        for j in range(total_batch_nums+1):
            image_batch_to_process = _get_this_batch(images_loaded_by_caffe, j, batch_size)
            num_images_being_processed = len(image_batch_to_process)
            data_blob_index = range(num_images_being_processed)
            # note that each batch is passed through a transformer before passing to data layer
            net.blobs['data'].data[data_blob_index] = [transformer.preprocess('data', img) for img in image_batch_to_process]
            # BEWARE: blobs arrays are overwritten
            res = net.forward()
            # actual batch feature extraction
            features_for_this_batch = net.blobs[extract_from_layer].data[data_blob_index].copy()
            features_all_images.extend(features_for_this_batch)
        
        #print(features_all_images)
        
        # store generated features in a binarized pickle file and write to disk
        pkl_object = {"filename": image_paths_list, "features": features_all_images}

        # TRYING TO PUT TO TEXT FILE TO SEE CAN DELETE LATER AS WE WILL USE PICKLE FILES
        self._save_features_to_file(features_all_images , image_paths_list , pretrained_model , extract_from_layer )

        output = open(output_pkl_file_name, 'wb')
        pickle.dump(pkl_object, output, 2)
        output.close()



    def _save_features_to_file(self , features , associated_filenames , modelname , layername):    
        for index , value in enumerate(features):
            #cwd = os.getcwd()   
            cwd = os.path.normpath(os.getcwd() + os.sep + os.pardir)   # one step back from getcwd()  
            # Note we need to work on what model and what layers to save. The following line is subject to change.
            # Here we create correct folder structure to save feature vector
            # Also strip out the "models" folder from the modelname
            modelname = os.path.basename(modelname)

            #replace / if present in layername by dash (-)
            layername = layername.replace("/" , "-")

            features_folder_path = os.path.join(cwd , "dataset", "features_etd1a" , modelname , layername )
            pathlib2.Path(features_folder_path).mkdir(parents=True, exist_ok=True)

            filename = os.path.splitext(associated_filenames[index])[0] + '.txt'
            filename = os.path.join(features_folder_path , os.path.basename(filename))
            print("So what is my new filename" , filename)
            np.savetxt(filename, value , newline="\n")

if __name__ == "__main__":
    obj_fe = FeatureExtraction()
    # Param 1 : Takes model name from EnumModels.
    # Param 2 : expected that models is the folder that contains this model and prototxt file. 
    #           But if it doesnot exist it will create a folder models and download necessary files
    obj_fe.extract_features("bvlc_alexnet" , "models")
    obj_fe.extract_features("bvlc_googlenet" , "models")




