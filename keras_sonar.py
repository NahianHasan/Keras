import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#fix  random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]


#encode class_values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)


def create_baseline_model():
	model = Sequential()
	model.add(Dense(60,input_dim=60, init='normal', activation='relu'))
	model.add(Dense(1,init='normal', activation = 'sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

estimator = KerasClassifier(build_fn=create_baseline_model, nb_epoch=500, batch_size=5)
#preparing cross_validation
kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

#Evalute the neural network
results = cross_val_score(estimator,X,encoder_Y, cv=kfold)
print("Accuracy: %0.2f%% (%0.2f%%)" % (results.mean()*100, results.std()*100))


