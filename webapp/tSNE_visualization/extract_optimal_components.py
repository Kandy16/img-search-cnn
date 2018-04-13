

import pickle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
rc={"lines.linewidth": 2.5})
matplotlib.use('Agg')
plt.switch_backend('agg')


## input file path
## output pickle dictionary
def loadPickleFile(filePath):
    print ("loading file....")
    pickleDict=pd.read_pickle("/var/www/img-search-cnn/webapp/dataset/KNN/bvlc_alexnet/fc7/vectors.p")
    print ("loading file finished.....")
    return pickleDict
	
	
	
## input dictionary of the image feature vectors along with the file name	
## output a tuple of ImageFeature Vectors in image_dataFrame, and the class and file_Name in Y dataFrame
def createFeatureVectorsDataFrame(featureDict):
    print ("creating pandas dataframe")
    print("keys ka lenght",len(featureDict.keys()))
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
	

	
def performPCA(features):
    print("performing PCA on dataFrame..")
    print("******************Shape of the image_dataFrame before PCA:",features.shape)
    ##svd = TruncatedSVD(n_components=no_of_dims,random_state=0)
    ##X_reduced=svd.fit_transform(features.values)
    feature_vect=StandardScaler().fit_transform(features.values)
    pca = PCA()
    pca_result = pca.fit_transform(feature_vect)
    print("******************Shape of the image_dataFrame after PCA:",pca_result.shape)
    return (pca,pca_result)
	
	
def drawPlotOfComponents(ca):
	##f = plt.figure(figsize=(9,8))
	plt.semilogy(ca.explained_variance_ratio_.cumsum(), '--o', label = 'Cumulative explained variance ratio')
	plt.xlabel('Principle components')
	plt.ylabel('Normalized proportion of data')
	plt.title('Variance represented by the principle components')
	plt.legend(loc='right')
	plt.savefig('/var/www/img-search-cnn/webapp/tSNE_visualization/plots/pca_components.png', dpi=120)
	
	
def sampleImageData(imageDF,YDF):
	sampled_image_dataFrame=imageDF.sample(frac=0.5)
	sampled_Y_dataFrame=YDF.loc[sampled_image_dataFrame.index]
	return (sampled_image_dataFrame,sampled_Y_dataFrame)
	
	
pickleDict=loadPickleFile('test')
image_dataFrame,Y=createFeatureVectorsDataFrame(pickleDict)
sampleImageDF,sampleYDF=sampleImageData(image_dataFrame,Y)
ca,pca_reduced=performPCA(sampleImageDF)
drawPlotOfComponents(ca)
