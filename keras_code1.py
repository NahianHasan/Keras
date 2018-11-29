from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("/home/nahian/Desktop/RATUL/Programming/Keras_coding/Databases/pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

print X,Y

#Split the dataset to train and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state = seed)


#Define Model
model = Sequential()
model.add(Dense(12,input_dim = 8, init = 'uniform', activation = 'relu'))
model.add(Dense(8,init = 'uniform', activation = 'relu'))
model.add(Dense(1,init = 'uniform', activation = 'sigmoid'))

#Compile Model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Fit the model
#model.fit(X,Y,validation_split = 0.33, nb_epoch = 300, batch_size = 768)
model.fit(X_train,Y_train,validation_data = (X_test,Y_test), nb_epoch = 300, batch_size = 768)

#Evaluate the model
scores = model.evaluate(X,Y)
print("%s: %0.2f" % (model.metrics_names[1], scores[1]*100))
