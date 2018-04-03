import pickle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import TruncatedSVD
import seaborn as sns




## input file path
## output pickle dictionary
def loadPickleFile(filePath):
	pickleDict=pd.read_pickle("/var/www/img-search-cnn/webapp/dataset/KNN/bvlc_alexnet/fc8/vectors.p")
	return pickleDict
	
	

	
loadPickleFile()



'''cols=None
	for key,value in featureDict.items():
		print "key::", key
		print "values::", value.shape
		cols=featureDict[key].shape[0]
		break
	print 'thesee many columns for dataFrame....',cols
	image_dataFrame=pd.DataFrame(columns=np.array(range(0,cols)))
	Y=pd.DataFrame(columns=['class','file_name'])
	
	i=0
	
	for key,value in featureDict.items():
		##if i==50:
			##break
		Y.loc[i]=[(float(key[0:3])),str(key)]
		##Y=Y.append({'class':(float(key[0:3])),'file_name':str(key)},ignore_index=True)
		image_dataFrame.loc[i]=value
		i=i+1
	'''