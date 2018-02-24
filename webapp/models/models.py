from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class Feedback(db.Model):
    __tablename__ = 'feedbacks'
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.String(64), unique=True)
    feature_vector = db.Column(db.String(255))

    def __str__(self):
        return self.query


class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    image_url = db.Column(db.String(128))
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedbacks.id'))
    feedback = db.relationship('Feedback', backref='feedback')

    def __str__(self):
        return self.image_url
