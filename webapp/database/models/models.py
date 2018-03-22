from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base


db = SQLAlchemy()
Base = declarative_base()

class Feedback(Base):
    __tablename__ = 'feedbacks'
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.String(64), unique=True)
    feature_vector = db.Column(db.String(255))
    feedback = db.relationship('Image', backref='feedback')
    def __str__(self):
        return self.query


class Image(Base):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    image_url = db.Column(db.String(128))
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedbacks.id'))

    def __str__(self):
        return self.image_url

class NeuralNetworkModel(Base):
    __tablename__ = 'neuralmodels'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    value = db.Column(db.String(50))
    neural_network = db.relationship('NeuralLayer', backref='neural_network')

    def __str__(self):
        return self.name

class NeuralLayer(Base):
    __tablename__ = 'neurallayer'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30))
    extracted = db.Column(db.Boolean, default=False)
    neural_network_id = db.Column(db.Integer, db.ForeignKey('neuralmodels.id'))
    mlalgorithms = db.relationship('MachineLearningAlgorithm', backref='mlalgorithms')

    def __str__(self):
        return self.name

class MachineLearningAlgorithm(Base):
    __tablename__ = 'mlalgorithm'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30))
    prepared = db.Column(db.Boolean, default=False)
    mlalgorithms_id = db.Column(db.Integer, db.ForeignKey('neurallayer.id'))

    def __str__(self):
        return self.name
