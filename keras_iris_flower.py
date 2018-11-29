import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#fix  random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load dataset
dataframe = pandas.read_csv("./Databases/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#encode class_values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)
#convert integers to dummy variables(i.e: one hot encoding)
dummy_Y = np_utils.to_categorical(encoder_Y)
print dummy_Y


def baseline_model():
	model = Sequential()
	model.add(Dense(4,input_dim=4, init='normal', activation='relu'))
	model.add(Dense(100, init='normal', activation='relu'))
	model.add(Dense(3,init='normal', activation = 'sigmoid'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=500, batch_size=5)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)#preparing cross_validation
#kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

#Evalute the neural network
results = cross_val_score(estimator,X,dummy_Y, cv=kfold)
print("Accuracy: %0.2f%% (%0.2f%%)" % (results.mean()*100, results.std()*100))












