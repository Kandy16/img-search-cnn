import pickle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib 
import random
matplotlib.use('Agg')
plt.switch_backend('agg')

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
rc={"lines.linewidth": 2.5})


## input file path
## output tuple of imgVectors and Ylabels
	
def loadfiles(filePath):
    imgVectors=filePath+'/transformedVectors.pkl'
    with open(imgVectors, "rb") as input_file:
        e = pickle.load(input_file)
        print 'the reduced image vectors shape::',e.shape
    Ylabels=pd.read_pickle(filePath+"/Y.pkl")
    print 'the loaded labels dataframe shape is :',Ylabels.shape
    return (e,Ylabels)

	

## input TSNE transformed_vectors and Ground truth Y to label them in scatterPlot
def drawScatterPlot(transformed_vectors,Ylabels,image_counter):
	print "drawing SCATTER PLOT on dataFrame values"
	
	##Creating the Unique classes found in the dataset
	uniqueCols=Ylabels['class'].unique()
	print "unique columns found::",len(uniqueCols)

	min=uniqueCols.min()
	max=uniqueCols.max()
	uniqueCols.sort()
	
	## randomly select 10 classes from the dataset
	ranCols=random.sample(range(int(min),int(max)), 10)
	ranCols = list(map(int,ranCols))
	print 'cols randomly selected::',len(ranCols)
	
	##convert the class column of the Ylable dataframe to int
	Ylabels['class']=Ylabels['class'].astype(int)
	
	## get data for the randomly selected columns:
	Ylabels['class']=Ylabels['class'].astype(int)
	shortDataFrame= Ylabels[Ylabels['class'].isin(ranCols)]
	
	print 'the shape of data frame containing 10 randomly selected classes::', shortDataFrame.shape
	palette = np.array(sns.color_palette("hls", len(ranCols)))
	colorArray=[]
	
	classColorDict={}
	for i in enumerate(ranCols):
		classColorDict[int(i[1])]=palette[i[0]]
	
	for i,row in shortDataFrame.iterrows():
		val=int(row['class'])
		colorArray.append(classColorDict[val])
		
	
	print "Found color for rows:::",len(colorArray)
	
	plt.scatter(transformed_vectors[shortDataFrame.index,0], transformed_vectors[shortDataFrame.index,1], lw=0, s=40,c=colorArray)
	plt.title("t-SNE generated plot")
	plt.xlabel('pc1')
	plt.ylabel('pc2')
	##ax.axis('off')
	##ax.axis('tight')
	##plt.show()
	plt.savefig('/var/www/img-search-cnn/webapp/tSNE_visualization/plots/tsne_generated_'+str(image_counter)+'.png', dpi=120)
	
file_path="/var/www/img-search-cnn/webapp/dataset/tSNE_visualization/features/pkl/finetune_flickr_style/fc8_flickr"

tVectors,sYDF=loadfiles(file_path)
drawScatterPlot(tVectors,sYDF,0)



