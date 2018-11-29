from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
import numpy


#Define Model
def create_model():
	model = Sequential()
	model.add(Dense(12,input_dim = 8, init = 'uniform', activation = 'relu'))
	model.add(Dense(8,init = 'uniform', activation = 'relu'))
	model.add(Dense(1,init = 'uniform', activation = 'sigmoid'))

	#Compile Model
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

#create model
model = KerasClassifier(build_fn = create_model, nb_epoch = 150, batch_size = 100, verbose = 1)


#Evaluate the model using 10-fold cross validation 
kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
results = cross_val_score(model,X,Y, cv=kfold)
print(results.mean())
