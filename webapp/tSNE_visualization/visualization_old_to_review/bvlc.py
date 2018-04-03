import pickle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import TruncatedSVD
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
rc={"lines.linewidth": 2.5})


## input file path
## output pickle dictionary
def loadPickleFile(filePath):
	print "loading file...."
	pickleDict=pd.read_pickle("/var/www/img-search-cnn/webapp/dataset/KNN/bvlc_alexnet/fc7/vectors.p")
	print "loading file finished....."
	return pickleDict
	
	
	
## input dictionary of the image feature vectors along with the file name	
## output a tuple of ImageFeature Vectors in image_dataFrame, and the class and file_Name in Y dataFrame
def	createFeatureVectorsDataFrame(featureDict):
	print "creating pandas dataframe"
	cols=None
	for key,value in featureDict.items():
		cols=featureDict[key].shape[0]
		break
	print cols
	image_dataFrame=pd.DataFrame(index=None,columns=np.array(range(0,cols)))
	Y=pd.DataFrame(index=None,columns=['class','file_name'])
	
	i=0
	
	for key,value in featureDict.items():
		if i==50:
			break
		Y=Y.append({'class':(float(key[0:3])),'file_name':str(key)},ignore_index=True)
		image_dataFrame=image_dataFrame.append(pd.Series(value),ignore_index=True)
		i=i+1
	print "creation of pandas dataframe finished.."
	return (image_dataFrame,Y)
	
	

## input the image_feature vectors dataframe and the number of dimensions
## output reduced image_vectors
def performSVD(features,no_of_dims):
	print "performing SVD on dataFrame.."
	
	print "******************Shape of the image_dataFrame before SVD:",image_dataFrame.shape
	svd = TruncatedSVD(n_components=no_of_dims,random_state=0)
	X_reduced=svd.fit_transform(image_dataFrame.values)
	print "******************Shape of the image_dataFrame after SVD:",image_dataFrame.shape
	return X_reduced
	
	
## input reduced dataframe, preplexity, and number of dimensions:
##	output n dimensionally transformed_vectors
def performTSNE(reducedDF,perplexity,no_of_dims):

	print "performing TSNE on dataFrame.."
	transformed_vectors=TSNE(n_components=no_of_dims,perplexity=perplexity).fit_transform(reducedDF)
	return transformed_vectors


## input TSNE transformed_vectors and Ground truth Y to label them in scatterPlot
def drawScatterPlot(transformed_vectors,Y,image_counter):
	print "drawing SCATTER PLOT on dataFrame values"
	
	##Creating the Unique classes found in the dataset
	uniqueCols=Y['class'].unique()
	uniqueCols.sort()
	palette = np.array(sns.color_palette("hls", 100))
	colorArray=[]
	for i in range(0,Y.shape[0]):
		colorArray.append(palette[int(Y.iloc[i][0])])
	
	print "Found these many colors:::",len(colorArray)
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	colors= Y.values
	sc = ax.scatter(transformed_vectors[:,0], transformed_vectors[:,1], lw=0, s=40,c=colorArray)
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')
	plt.show()
	plt.savefig('/var/tmp/tsne_generated_'+image_counter+'_.png', dpi=120)
	

	
pickleDict=loadPickleFile('test')
image_dataFrame,Y=createFeatureVectorsDataFrame(pickleDict)
X_reduced=performSVD(image_dataFrame,30)

for i in enumerate([5,10]):
	transformed_vectors=performTSNE(X_reduced,i[1],2)
	drawScatterPlot(transformed_vectors,Y,i[0])



