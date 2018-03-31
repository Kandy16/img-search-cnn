import os , glob
from ..ml.cosine import cosine_similarity_cluster as cs 
obj_cosine = cs.CosineSimilarityCluster() # object of KNN used for extract , feedback

# First we need to get 5 random images from each of the evaluation images section. note the images are of format 00
class EvaluationB(object):
	def __init__(self, image_location , knn_save_path, cosine_nearest_save_path):
		self.image_location = image_location
		self.knn_save_path = knn_save_path
		self.cosine_nearest_save_path = cosine_nearest_save_path
		self.ground_truth = self._get_ground_truth()

	def evaluate(self, modelname, layername):
		# First we get random images from each category. i.e 5 images from each folders.
		random_samples = self._get_random_sample(5)

		# loop through each of this random sample images
		# get the feedback given by the certain model
		# check how many of those elements fall under the ground truth folder val
		calculated_cosine_neighbours_path = os.path.join("/var/www/img-search-cnn/webapp/dataset/COSINE" , "bvlc_alexnet" , "fc7")
		for sample in random_samples[:1]:
			first_three_digit = sample[:3]  # determines which dictionary to look at for ground truth
			rand_images = obj_cosine.get_feedback(calculated_cosine_neighbours_path , ["001_0016.jpg"])
			ground_truth = self.ground_truth[first_three_digit]

			# now we compare how many rand_images fall in the ground truth
			print([i in ground_truth for i in rand_images])
		#print(rand_images)


	def _get_image_list_from_files(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))

		filename = self.image_location
		files_grabbed = []
		if os.path.exists(filename):
			os.chdir(filename)

			types = ('*.jpeg', '*.jpg' , '*.JPEG' , '*.png') # the tuple of file types
			
			for files in types:
				files_grabbed.extend(glob.glob(files))
			os.chdir(dir_path)
		else:
			print("No Folder exist of name \"" +  self.image_location + "\" Please create and put images into it")
		return files_grabbed

	def _get_ground_truth(self):
		from itertools import groupby , chain
		import random
		tags = self._get_image_list_from_files()
		m = {i:list(j) for i , j in groupby(sorted(tags), key=lambda x: x[:3])}
		return m

	# Gets random images (n) from each cluster 
	# n = 5 means 5 images from each folder or cluster
	def _get_random_sample(self, n):
		from itertools import groupby , chain
		import random
		tags = self._get_image_list_from_files()
		m = [list(j) for i , j in groupby(sorted(tags), key=lambda x: x[:3])]
		lists = [random.sample(i, n if len(i) >= n else len(i)) for i in m]
		random_samples = list(chain(*lists))
		return random_samples


if __name__ == "__main__":
	img_path = "/var/www/img-search-cnn/webapp/dataset/images_eval"
	obj_eval_b = EvaluationB(img_path , "" , ""  )
	
	#print(obj_eval_b.get_random_sample(5 , img_path))
	#n = obj_eval_b._get_ground_truth()
	#print(n["001"])
	obj_eval_b.evaluate("", "")