import numpy as np, random, math, os
from PIL import Image
import time
from image_clustering import ImageClustering


class RandomImages(object):
    def __init__(self, vectors_save_location , clusters_save_location , modelname , layername, number_of_clusters=10):
        self.number_of_clusters = number_of_clusters
        self.vectors_save_location = vectors_save_location
        self.clusters_save_location = clusters_save_location
        self.modelname = modelname
        self.layername = layername

        # Clusters has to be built for some functions below
        self.obj_ic = ImageClustering(number_of_clusters)
       
    # Helper function for get_n_random_images_full_random. Loads all feature_vectors file and creates given image format filename. 
    def _get_filenames_list(self, images_location):
        files = []
        for file in os.listdir(images_location):
            files.append(file)
        return files    
      
    def get_n_random_images_full_random(self, images_location, n):
        random_images = []
        files = self._get_filenames_list(images_location)
        for x in range(0, n):
            filename = random.choice(files)
            random_images.append(filename)
        return random_images

    def get_n_random_images_from_clusters(self, n , image_format):
        random_images = []
        clusters , _ = self.obj_ic.get_clusters(self.vectors_save_location , self.clusters_save_location , self.modelname , self.layername)
        for x in range(0, n):
            filename = random.choice(clusters[x % self.number_of_clusters])
            filename = os.path.splitext(filename)[0] + image_format # format expected as .jpg or others
            random_images.append(filename)
        return random_images

    # draws n samples from a given probability distribution
    def get_n_random_images_based_on_feedback(self, n, image_format, positive_feedback, negative_feedback):
        clusters , reversed_clusters = self.obj_ic.get_clusters(self.vectors_save_location , self.clusters_save_location , self.modelname , self.layername)
        probabilities, probabilities_accumulated = self._compute_probabilties(positive_feedback, negative_feedback , reversed_clusters)
        random_images = []
        for x in range(0, n):
            random_cluster = random.uniform(0, 1)
            for index_of_cluster in range(0, len(probabilities_accumulated)):
                if probabilities_accumulated[index_of_cluster] > random_cluster:
                    random_cluster = index_of_cluster
                    break
            filename = random.choice(clusters[random_cluster])
            filename = os.path.splitext(filename)[0] + image_format # format expected as .jpg or others
            random_images.append(filename)
        return random_images

    # adjusts the Probablities based on feedback
    def _compute_probabilties(self, relevant_images, irrelevant_images , reversed_clusters):        
        probabilities = []
        irrelevant_part_probabilities = []
        for index_of_cluster in range(0, self.number_of_clusters):
            counter_irrelevant = 0
            for image in irrelevant_images:
                if reversed_clusters[image] == index_of_cluster:
                    counter_irrelevant += 1
            irrelevant_part_temp = 1/(math.pow(100, counter_irrelevant))
            irrelevant_part_probabilities.append(irrelevant_part_temp)
        sum = 0
        for irrelevant_part_temp in irrelevant_part_probabilities:
            sum += irrelevant_part_temp
        for index_of_cluster in range(0, self.number_of_clusters):
            probability = irrelevant_part_probabilities[index_of_cluster] / sum
            probabilities.append(probability)

        # accumalate probabilties
        probabilities_accumalated = []
        probabilities_accumalated.append(probabilities[0])
        # print(probabilities[0])
        for x in range(1, self.number_of_clusters):
            probabilities_accumalated.append(probabilities_accumalated[(x - 1)] + probabilities[x])
        return probabilities, probabilities_accumalated

    # # adjusts the Probablities based on feedback
    # def compute_probabilties_old(self, relevant_images, irrelevant_images):
    #     # 10 % der Bilder komplett Random, 70% basiert auf relevanten Bildern, 20% basiert auf irrelevanten
    #     # 0.1/number_of_clusters + (number_of_relevant_in_cluster-number_of_irrelevant_in_cluster)/number_of_relevant_images
    #     # todo decide how to store relevant and irrelevant images
    #     # relevant_images = ['043785.txt', '012723.txt']
    #     # irrelevant_images = ['058987.txt', '014541.txt', '045068.txt', '080452.txt']
    #     # relevant_images = []
    #     # irrelevant_images = []
    #     relevant_part = 0.7
    #     if len(relevant_images) == 0:
    #         relevant_part = 0
    #     irrelevant_part = 0.9 - relevant_part
    #     if len(irrelevant_images) == 0:
    #         irrelevant_part = 0
    #     random_part = 1 - irrelevant_part - relevant_part


    #     probabilities = []
    #     relevant_part_probablities = []
    #     irrelevant_part_probabilities = []
    #     for index_of_cluster in range(0, self.number_of_clusters):
    #         counter_relevant = 0
    #         for image in relevant_images:
    #             if self.reversed_clusters[image] == index_of_cluster:
    #                 counter_relevant += 1
    #         counter_irrelevant = 0
    #         for image in irrelevant_images:
    #             if self.reversed_clusters[image] == index_of_cluster:
    #                 counter_irrelevant += 1
    #         if len(relevant_images) != 0:
    #             relevant_part_probablities.append(relevant_part * counter_relevant/len(relevant_images))
    #             # relevant_part_temp = relevant_part * counter_relevant/len(relevant_images)
    #         if len(irrelevant_images) != 0:
    #             irrelevant_part_temp = 1/(math.pow(10, counter_irrelevant))
    #             # irrelevant_part_temp = (irrelevant_part / self.number_of_clusters)
    #             irrelevant_part_probabilities.append(irrelevant_part_temp)
    #         # probability = relevant_part_temp + irrelevant_part_temp + random_part/self.number_of_clusters
    #     sum = 0
    #     for irrelevant_part_temp in irrelevant_part_probabilities:
    #         sum += irrelevant_part_temp
    #     for index_of_cluster in range(0, self.number_of_clusters):
    #         probability = random_part/self.number_of_clusters
    #         if len(relevant_images) != 0:
    #             probability += relevant_part_probablities[index_of_cluster]
    #         if len(irrelevant_images) != 0:
    #             probability += irrelevant_part * (irrelevant_part_probabilities[index_of_cluster] / sum)
    #         probabilities.append(probability)

    #     probabilities_accumalated = []
    #     probabilities_accumalated.append(probabilities[0])
    #     # print(probabilities[0])
    #     for x in range(1, self.number_of_clusters):
    #         probabilities_accumalated.append(probabilities_accumalated[(x - 1)] + probabilities[x])
    #     return probabilities, probabilities_accumalated


if __name__ == "__main__":
    extract_info = {"model_name":"finetune_flickr_style" , "model_layer":"fc8_flickr"}
    vectors_save_location = "/var/www/img-search-cnn/webapp/dataset/KNN"
    clusters_save_location = "/var/www/img-search-cnn/webapp/dataset/CLUSTERING"
    images_location = "/var/www/img-search-cnn/webapp/dataset/images"
    #replace / if present in layername by dash (-)
    smoothed_layer_name = extract_info["model_layer"].replace("/" , "-")

    obj = RandomImages(vectors_save_location , clusters_save_location , extract_info["model_name"] , smoothed_layer_name, 10)
    print("Full Random : " , obj.get_n_random_images_full_random(images_location , 20))
    print("Using clusters : " , obj.get_n_random_images_from_clusters(20 , ".jpg"))
    print("Using clusters with feedbacks: " , obj.get_n_random_images_based_on_feedback(20 , ".jpg" , [] , []))
    #random_images = obj.get_n_random_images_based_on_feedback(20)