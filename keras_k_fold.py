from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

#kfold cross validation test_harness
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state =seed)
cvscores = []

for train,test in kfold.split(X,Y):
	
	#Define Model
	model = Sequential()
	model.add(Dense(12,input_dim = 8, init = 'uniform', activation = 'relu'))
	model.add(Dense(8,init = 'uniform', activation = 'relu'))
	model.add(Dense(1,init = 'uniform', activation = 'sigmoid'))

	#Compile Model
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	#Fit the model
	#model.fit(X,Y,validation_split = 0.33, nb_epoch = 300, batch_size = 768)
	model.fit(X[train], Y[train], nb_epoch = 300, batch_size = 768, verbose = 0)

	#Evaluate the model
	scores = model.evaluate(X[test],Y[test])
	print("%s: %0.2f" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
