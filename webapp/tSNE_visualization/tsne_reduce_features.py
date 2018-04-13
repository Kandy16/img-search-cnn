import pickle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import time
from sklearn.decomposition import TruncatedSVD
import pdb
import os
import pathlib2

class TSNEReduceFeatures(object):
	def __init__(self, vectors_p_path ):
		self.vectors_p_path = vectors_p_path

	## takes vectors
	## output pickle dictionary
	def loadPickleFile(self, filePath):
		print "loading file from:::",filePath
		pickleDict=pd.read_pickle(filePath)
		print "loading file finished....."
		return pickleDict

	
	## input dictionary of the image feature vectors along with the file name	
	## output a tuple of ImageFeature Vectors in image_dataFrame, and the class and file_Name in Y dataFrame
	def	createFeatureVectorsDataFrame(self, featureDict):
		print "creating pandas dataframe"
		print "keys ka lenght",len(featureDict.keys())
		image_dataFrame=pd.DataFrame()
		image_dataFrame=image_dataFrame.from_dict(featureDict)
		image_dataFrame=image_dataFrame.T
		image_dataFrame['file_name']=image_dataFrame.index
		image_dataFrame.reset_index(drop=True,inplace=True)
		image_class= image_dataFrame['file_name'].tolist()
		image_class_list=[ x[0:3] for x in image_class]
		image_dataFrame['class']=image_class_list
		Y=pd.concat([image_dataFrame['file_name'],image_dataFrame['class']],axis=1)
		image_dataFrame.drop(['file_name','class'],axis=1,inplace=True)
		print("creation of pandas dataframe finished..")
		return (image_dataFrame,Y)	
	

	## input the image_feature vectors dataframe and the number of dimensions
	## output reduced image_vectors
	def performSVD(self, features,no_of_dims):
		print "performing SVD on dataFrame.."
		
		print "******************Shape of the image_dataFrame before SVD:",features.shape
		svd = TruncatedSVD(n_components=no_of_dims,random_state=0)
		X_reduced=svd.fit_transform(features.values)
		print "******************Shape of the image_dataFrame after SVD:",X_reduced.shape
		return X_reduced
		
		
	## input reduced dataframe, preplexity, and number of dimensions:
	##	output n dimensionally transformed_vectors
	def performTSNE(self, reducedDF,perplexity,no_of_dims):

		print "performing TSNE on dataFrame.."
		transformed_vectors=TSNE(n_components=no_of_dims,perplexity=perplexity).fit_transform(reducedDF)
		return transformed_vectors	


	def sampleImageData(self, imageDF,YDF):
		sampled_image_dataFrame=imageDF.sample(frac=1)
		sampled_Y_dataFrame=YDF.loc[sampled_image_dataFrame.index]
		print 'sampled_image_dataFrame shape',sampled_image_dataFrame.shape
		print 'sampled_Y_dataFrame shape',sampled_Y_dataFrame.shape
		
		return (sampled_image_dataFrame,sampled_Y_dataFrame)

	def saveReducedVectors(self, transVectors,sampleYDF,feature_save_path , modelname , layername):
		print 'saving the dimensionally reduced vectors to file....'
		file_path=feature_save_path
		sampleYDF['original_index']=sampleYDF.index
		sampleYDF.reset_index(drop=True,inplace=True)


		pkl_save_loc = os.path.join(file_path ,"pkl", modelname , layername)
		pathlib2.Path(pkl_save_loc).mkdir(parents=True, exist_ok=True)

		sampleYDF.to_pickle(os.path.join(pkl_save_loc , "Y.pkl"))

		with open(os.path.join(pkl_save_loc , "transformedVectors.pkl"),'wb') as output:
			pickle.dump(transVectors,output,pickle.HIGHEST_PROTOCOL)
			
		print 'transVectors shape',transVectors.shape
		print 'sampled_Y_dataFrame shape',sampleYDF.shape
		
		reduced_feature_save_loc = os.path.join(file_path ,"reduced_features", modelname , layername)		
		pathlib2.Path(reduced_feature_save_loc).mkdir(parents=True, exist_ok=True)

		for i,row in enumerate(transVectors):
			np.savetxt(os.path.join(reduced_feature_save_loc , sampleYDF.loc[i]['file_name']),row,newline="\n")		
		print 'save complete!!!'


	def perform_feature_reduction(self , feature_save_path, modelname , layername):
		file_path_name = os.path.join(self.vectors_p_path , modelname , layername, "vectors.p")
		if os.path.exists(file_path_name):
			pickle_dict=self.loadPickleFile(file_path_name)
			image_dataFrame,Y=self.createFeatureVectorsDataFrame(pickle_dict)
			sampleImageDF,sampleYDF=self.sampleImageData(image_dataFrame,Y)
			X_reduced=self.performSVD(sampleImageDF,10)
			transformed_vectors=self.performTSNE(X_reduced,5,2)
			self.saveReducedVectors(transformed_vectors, sampleYDF, feature_save_path , modelname , layername)
		else:
			print("No vectors.p file can be loaded from the location : " , file_path_name)

if __name__ == "__main__":
	vectors_p_path = "/var/www/img-search-cnn/webapp/dataset/KNN"
	feature_save_path = "/var/www/img-search-cnn/webapp/dataset/tSNE_visualization/features"
	#modelname = "finetune_flickr_style"
	#layername = "fc8_flickr"

	#modelname = "finetune_flickr_style"
	#layername = "fc7"

	#modelname = "ResNet18_ImageNet_CNTK_model"
	#layername = "z"

	#modelname = "bvlc_reference_caffenet"
	#layername = "fc7"

	modelname = "bvlc_reference_caffenet"
	layername = "fc8"

	obj_TSNE1 = TSNEReduceFeatures(vectors_p_path)
	obj_TSNE1.perform_feature_reduction(feature_save_path, modelname , layername)
	
	modelname = "bvlc_alexnet"
	layername = "fc7"
	obj_TSNE2 = TSNEReduceFeatures(vectors_p_path)
	obj_TSNE2.perform_feature_reduction(feature_save_path, modelname , layername)
	

	modelname = "bvlc_alexnet"
	layername = "fc8"
	obj_TSNE3 = TSNEReduceFeatures(vectors_p_path)
	obj_TSNE3.perform_feature_reduction(feature_save_path, modelname , layername)
	

	modelname = "bvlc_googlenet"
	layername = "pool5-7x7_s1"
	obj_TSNE4 = TSNEReduceFeatures(vectors_p_path)
	obj_TSNE4.perform_feature_reduction(feature_save_path, modelname , layername)
	

