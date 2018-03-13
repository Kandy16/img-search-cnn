from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np
from ast import literal_eval

class CosineSimilarityCluster(object):
  def __init__(self  , imagevectors_filepath= "/var/www/clone-img-search-cnn/img-search-cnn/webapp/dataset/features/fc8", dimension = 1000 , n_nearest_neighbors = 10 , trees= 100):
    # data structures
    self.file_index_to_file_name = {}
    self.file_index_to_file_vector = {}
    self.chart_image_positions = {}
    self.imagevectors_filepath = imagevectors_filepath
    self.n_nearest_neighbors = n_nearest_neighbors
    self.dimension = dimension
    self.trees = trees

  def nearest_neighbours_for_each_imagevector(self):        
    # config
    dims = self.dimension
    n_nearest_neighbors = self.n_nearest_neighbors
    trees = self.trees
    infiles = glob.glob(self.imagevectors_filepath + '/*.txt')

    # build ann index
    t = AnnoyIndex(dims)
    kkk = 0
    for file_index, i in enumerate(infiles):
      kkk = kkk + 1
      print("/n" , kkk)
      file_vector = np.loadtxt(i)
      file_name = os.path.basename(i).split('.')[0]
      self.file_index_to_file_name[file_index] = file_name
      self.file_index_to_file_vector[file_index] = file_vector
      t.add_item(file_index, file_vector)
    t.build(trees)
    print(os.getcwd())
    if not os.path.exists(self.imagevectors_filepath):
      print("No such file exits where we can load the image vectors from")
    else:      
      # create a nearest neighbors json file for each input
      if not os.path.exists('cosine_nearest_neighbors'):
        os.makedirs('cosine_nearest_neighbors')

      for i in self.file_index_to_file_name.keys():
        master_file_name = self.file_index_to_file_name[i]
        master_vector = self.file_index_to_file_vector[i]

        named_nearest_neighbors = []
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
        for j in nearest_neighbors:
          neighbor_file_name = self.file_index_to_file_name[j]
          neighbor_file_vector = self.file_index_to_file_vector[j]

          similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
          rounded_similarity = int((similarity * 10000)) / 10000.0

          named_nearest_neighbors.append({
            'filename': neighbor_file_name,
            'similarity': rounded_similarity
          })

        with open('cosine_nearest_neighbors/' + master_file_name + '.json', 'w') as out:
          json.dump(named_nearest_neighbors, out)


  def get_filenames_cosine_neighbour(self, filepath):
    with open(filepath) as f:
        for line in f:  # note there will just be one line in the file i.e list of dictionaries
          mainlist = list(literal_eval(line))
    return [i['filename'] + '.jpg' for i in mainlist]

if __name__ == "__main__":
  obj = CosineSimilarityCluster()
  obj.nearest_neighbours_for_each_imagevector()
#
# using filenames from neighbours json file test
#print(get_filenames_cosine_neighbour('dup_cosine_nearest_neighbors/014999.json'))