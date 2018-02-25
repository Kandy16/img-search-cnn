from PIL import Image
import pickle
import time as time
import numpy as np
from random import randint
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA


#This is loading the features vectors which were stored by the KNN.py file
images = pickle.load(open("vectors10000.p", "rb"))



#This stuff is preparing the data so that the clustering algorithm can use it
imagesNew = []
imagesLabels = []
for x,y in images.items():
    imagesNew.append(y)
    imagesLabels.append(x)

dataSet = np.stack(imagesNew, axis=0)


#PCA will decrease the number of dimensions of our data, it will decrease it as much as possible but will keep 75% of the information from the data
pca = PCA(n_components=0.75, svd_solver='full')
pca.fit(dataSet)
dataSet = pca.transform(dataSet)
print('transforming done!')




# The following stuff is still quiet messy, I was trying to test different clustering algorithm performance and quality wise

st = time.time()


# This builds a connectivity matrix and can be used to improve AgglomerativeClustering Performance by a lot

# connectivity = kneighbors_graph(dataSet, n_neighbors=10, include_self=False)
# print('Connectivity Matrix computed!')
# ward = AgglomerativeClustering(n_clusters=1000, connectivity=connectivity, linkage='ward').fit(dataSet)



# Different Type of Clustering Algorithms with 1000 clusters on the data

ward = AgglomerativeClustering(n_clusters=1000, linkage='ward').fit(dataSet)
# ward = MiniBatchKMeans(n_clusters=10000, init_size=15000).fit(dataSet)
# ward = KMeans(n_clusters=1000).fit(dataSet)

elapsed_time = time.time() - st
clusterLabels = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)



# This loop will give you the possilibty to display different clusters from the 1000 clusters. Just enter a number between 1-1000 and you will be shown all images inside this cluster. You can exit this loop by entering -1
while True:
    cluster = int(input("Show cluster: "))
    if cluster == -1:
        break
    n=0
    for x in clusterLabels:
        if x == cluster:
            filename = imagesLabels[n]
            img = Image.open('media/images/' + filename[:-3] + 'jpg')
            time.sleep(0.1)
            img.show()
        n = n + 1




# Very early idea of using clustering to improve learning phase, you get shown 10 images and have to select the most relevant image by entering the filename of the image



# initial step of random images

print("Compute hierarchical clustering for dataset")
numberOfClusters = 10
st = time.time()
connectivity = kneighbors_graph(dataSet, n_neighbors=10, include_self=False)
# connectivity = pickle.load(open("connectivityMatrix10.p", "rb"))
# pickle.dump(connectivity, open("connectivityMatrix10.p", "wb"))
elapsed_time = time.time() - st
print("Elapsed time for connectivity: %.2fs" % elapsed_time)
ward = AgglomerativeClustering(n_clusters=numberOfClusters, connectivity=connectivity, linkage='ward').fit(dataSet)
elapsed_time = time.time() - st
print("Elapsed time: %.2fs" % elapsed_time)


n = 0
while n < numberOfClusters:
    random = randint(0, ward.labels_.size-1)
    if ward.labels_[random] == n:
        n = n+1
        filename = imagesLabels[random]
        img = Image.open('media/images/' + filename[:-3] + 'jpg')
        time.sleep(0.5)
        img.show()
        print(random)


while True:
    numberOfClusters = numberOfClusters * 2
    feedback = 0
    while True:
        feedback = int(input('Enter file number of relevant image:'))
        filename = imagesLabels[feedback]
        img = Image.open('media/images/' + filename[:-3] + 'jpg')
        time.sleep(0.5)
        img.show()
        confirm = input('Confirm? Yes/No')
        if confirm in ('Yes', 'yes', 'Y', 'y'):
            break
    random = randint(0, ward.labels_.size - 1)
    n = 0
    while n < 10:
        random = randint(0, ward.labels_.size - 1)
        if ward.labels_[random] == ward.labels_[feedback]:
            n = n + 1
            filename = imagesLabels[random]
            img = Image.open('media/images/' + filename[:-3] + 'jpg')
            time.sleep(0.5)
            img.show()
            print(random)
    # ward = AgglomerativeClustering(n_clusters=numberOfClusters, linkage='ward').fit(dataSet)
    ward = AgglomerativeClustering(n_clusters=numberOfClusters, connectivity=connectivity, linkage='ward').fit(dataSet)
    print('Clustering with ' + str(numberOfClusters) + ' number of clusters done!')