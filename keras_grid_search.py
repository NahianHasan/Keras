from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy


#Define Model
def create_model(optimizer = 'rmsprop', init='glorot_uniform'):
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

dataset = numpy.loadtxt("Databases/pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

#create model
model = KerasClassifier(build_fn = create_model, verbose = 1)

#grid Search Parameters
optimizers = ['rmsprop','adam']
init = ['glorot_uniform','uniform','normal']
epochs = numpy.array([50,100,150])
batches = numpy.array([5,10,20])
param_grid = dict(optimizer=optimizers,nb_epoch=epochs,batch_size = batches,init=init)
grid = GridSearchCV(estimator=model,param_grid=param_grid)
grid_result = grid.fit(X,Y)

#Summarize Results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params,mean_score,scores in grid_result.grid_scores_:
	print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
