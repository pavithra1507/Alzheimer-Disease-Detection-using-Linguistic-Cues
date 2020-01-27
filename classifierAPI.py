import flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf



# Initialize application and Keras model

app = flask.Flask(__name__)
api = Api(app)


def init():
	global model, tokenizer, graph
	model = load_model('models/classifier.h5')
	graph = tf.get_default_graph()
	with open('models/tokenizer.pkl', 'rb') as f:
    		tokenizer = pickle.load(f)


# parse argument

parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
	def get(self):
		args = parser.parse_args()
		user_query = args['query']
		uq_tokenized = tokenizer.texts_to_sequences(np.array([user_query]))
		uq_tokenized = pad_sequences(uq_tokenized, padding = 'post', maxlen = 100)
		with graph.as_default():
			prediction = model.predict(uq_tokenized)
			pred_proba = model.predict_proba(uq_tokenized)
       		# Output either 'Negative' or 'Positive' along with the score
		if prediction < 0.5:
			pred_text = 'Negative'
		else:
			pred_text = 'Positive'
        	# round the predict proba value and set to new variable
		# confidence = round(pred_proba[0], 3)
        	# create JSON object
		output = {'prediction': pred_text}
		return output

api.add_resource(PredictSentiment, '/')

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
	init()
	app.run(debug = True)