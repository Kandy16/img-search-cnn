from .models.models import NeuralNetworkModel , MachineLearningAlgorithm, NeuralLayer
from feature_extraction.EnumModels import Models

class Database(object):
	def __init__(self):
		pass

	def fillin_database(self,db):
		# Now we fill in the models using the ENUMMODELS
		for model in Models:
			print(model.value)
			obj_neural_model = NeuralNetworkModel(name=model.name , value = model.value)
			db.session.add(obj_neural_model)
			db.session.commit()

			if (model.name == Models.bvlc_alexnet.name or model.name == Models.bvlc_reference_caffenet.name):
				obj_neural_layer = NeuralLayer(name='fc7' , neural_network = obj_neural_model  , extracted= False)
				db.session.add(obj_neural_layer)
				obj_ml_knn = MachineLearningAlgorithm(name='knn' , prepared= False, mlalgorithms=obj_neural_layer)
				obj_ml_cos = MachineLearningAlgorithm(name='cosine' , prepared= False, mlalgorithms=obj_neural_layer)
				db.session.add(obj_ml_knn)
				db.session.add(obj_ml_cos)
				db.session.commit()

				obj_neural_layer = NeuralLayer(name='fc8' , neural_network = obj_neural_model  , extracted= False)
				db.session.add(obj_neural_layer)
				obj_ml_knn = MachineLearningAlgorithm(name='knn' , prepared= False, mlalgorithms=obj_neural_layer)
				obj_ml_cos = MachineLearningAlgorithm(name='cosine' , prepared= False, mlalgorithms=obj_neural_layer)
				db.session.add(obj_ml_knn)
				db.session.add(obj_ml_cos)
				db.session.commit()

			if (model.name == Models.bvlc_googlenet.name) :
				obj_neural_layer = NeuralLayer(name='pool5/7x7_s1' , neural_network = obj_neural_model  , extracted= False)
				db.session.add(obj_neural_layer)
				obj_ml_knn = MachineLearningAlgorithm(name='knn' , prepared= False, mlalgorithms=obj_neural_layer)
				obj_ml_cos = MachineLearningAlgorithm(name='cosine' , prepared= False, mlalgorithms=obj_neural_layer)
				db.session.add(obj_ml_knn)
				db.session.add(obj_ml_cos)
				db.session.commit()

			if (model.name == Models.finetune_flickr_style.name):
				obj_neural_layer = NeuralLayer(name='fc7' , neural_network = obj_neural_model  , extracted= False)
				db.session.add(obj_neural_layer)
				obj_ml_knn = MachineLearningAlgorithm(name='knn' , prepared= False, mlalgorithms=obj_neural_layer)
				obj_ml_cos = MachineLearningAlgorithm(name='cosine' , prepared= False, mlalgorithms=obj_neural_layer)
				db.session.add(obj_ml_knn)
				db.session.add(obj_ml_cos)
				db.session.commit()

				obj_neural_layer = NeuralLayer(name='fc8_flickr' , neural_network = obj_neural_model  , extracted= False)
				db.session.add(obj_neural_layer)
				obj_ml_knn = MachineLearningAlgorithm(name='knn' , prepared= False, mlalgorithms=obj_neural_layer)
				obj_ml_cos = MachineLearningAlgorithm(name='cosine' , prepared= False, mlalgorithms=obj_neural_layer)
				db.session.add(obj_ml_knn)
				db.session.add(obj_ml_cos)
				db.session.commit()

		db.session.commit()
