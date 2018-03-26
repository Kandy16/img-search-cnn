from webapp.ml.clustering.random_images import RandomImages
from webapp.ml.knn.knn import KNN


class EvalMetrics(object):

    def __init__(self, path_to_dataset, model_name, layer, output_dir):
        self.path_to_images_features = path_to_dataset + model_name + '/' + layer + '/'
        self.model_name = model_name
        self.layer = layer
        self.output_filename = output_dir + self.model_name + '_' + self.layer


    # This is preparing the data so that the clustering algorithm can use it
    def compute_precision(self, retrieved_images, keyword, iteration):
        counter = 0
        for x in retrieved_images:
            if x[:3] == keyword:
                counter += 1
        old = open(self.output_filename + '_eval_detail.txt', 'r')
        old = old.read()
        file = open(self.output_filename + '_eval_detail.txt', 'w')
        file.write(old + '\n')
        file.write(self.model_name + ' ' + self.layer + ' ' + str(keyword) + ' ' + str(iteration) + ' ' + str(counter/len(retrieved_images)))
        file.close()
        self.precision_values[keyword][iteration].append(counter/len(retrieved_images))
        return counter/len(retrieved_images)

    def run_tests(self, number_of_tests, iterations_to_examine, keywords):
        file = open(self.output_filename + '_eval_detail.txt', 'w')
        file.write('model_name layer keyword iteration precision')
        file.close()
        default_randomizer = RandomImages(path_to_features=self.path_to_images_features, number_of_clusters=10)
        randomizer = RandomImages(path_to_features=self.path_to_images_features, number_of_clusters=1000)
        self.precision_values = {}
        for keyword in keywords:
            self.precision_values[keyword] = {}
            for iteration in iterations_to_examine:
                self.precision_values[keyword][iteration] = []
            for x in range(0, number_of_tests):
                self.simulate_feedback(keyword, iterations_to_examine, default_randomizer, randomizer)
        self.summarize_results()
        self.result_first_hit(number_of_tests)


    def simulate_feedback(self, keyword, iterations_to_examine, randomizer_initial, randomizer):
        positive_feedback = []
        negative_feedback = []
        random_images = randomizer_initial.get_n_random_images(10)
        # random_images = randomizer_initial.get_n_random_images_full_random(10)
        retrieved_images = random_images

        for iteration in range(1, max(iterations_to_examine)+1):
            if iteration in iterations_to_examine:
                # results = ['001_0002', '001_0010', '001_0015', '001_0037', '002_0074', '004_0114', '005_0024', '005_0021']
                self.compute_precision(retrieved_images, keyword, iteration)
            for x in retrieved_images:
                if x[:3] == keyword:
                    if x not in positive_feedback:
                        positive_feedback.append(x)
                else:
                    if x not in negative_feedback:
                        negative_feedback.append(x)
            # retrieved_images = randomizer_initial.get_n_random_images(10)
            # retrieved_images = randomizer_initial.get_n_random_images_full_random(10)
            retrieved_images = randomizer.get_n_random_images_based_on_feedback(10, positive_feedback, negative_feedback)
            if len(positive_feedback) != 0:
                knn = KNN()
                new_images = knn._find_k_neighbours(10, randomizer.clustering.feature_vectors, randomizer.clustering.feature_vectors[positive_feedback[0]], 'euclidean')
                retrieved_images = []
                for x in new_images:
                    retrieved_images.append(x[1])
            # print(retrieved_images)
        return 0

    def summarize_results(self):
        file = open(self.output_filename + '_eval_summary.txt', 'w')
        file.write('model_name layer keyword iteration precision')

        for keyword, value in self.precision_values.items():
            for iteration, precision_values in self.precision_values[keyword].items():
                counter = 0
                for x in precision_values:
                    counter += x
                average_precision = counter/len(precision_values)
                file.write('\n')
                file.write(self.model_name + ' ' + self.layer + ' ' + str(keyword) + ' ' + str(iteration) + ' ' + str(average_precision))
        file.close()


    def result_first_hit(self, number_of_tests):
        file = open(self.output_filename + '_eval_first_hit.txt', 'w')
        file.write('Average number of iterations until first relevant image was found')

        counter = 0
        for x in range(0, number_of_tests):
            for keyword, value in self.precision_values.items():
                for iteration, precision_values in self.precision_values[keyword].items():
                    if precision_values[x] != 0:
                        counter += iteration
                        break
                else:
                    counter += 40
        file.write('\n')
        result = counter/(number_of_tests*25)
        file.write(str(result))
        file.close()






if __name__ == "__main__":
    classes_to_observe = []
    for x in range(1, 25):
        classes_to_observe.append("%03d" % (x,))

    eval = EvalMetrics('/Users/volk/Development/Dataset/features_etd1a_images_eval/', 'bvlc_alexnet', 'fc8', '/Users/volk/Development/Dataset/TESTING/')
    # eval.run_tests(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['001', '002', '003', '004', '005'])
    eval.run_tests(5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], classes_to_observe)