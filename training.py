# 3. Training and Evaluation
# 3.1 split data, use 20%
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
print('training has %d observation with %d features'% X_train.shape)
print('test data has %d observation with %d features'% X_test.shape)

''' scale the data using 
1. standardization (x-mean)/std
2. normalization (x - x_min)/(x_max - x_min)
reason:
1. speed up gradient descent
2. same scale
for example, use training data to train the standardscaler to get mean and std
apply mean and std to both training and testing data
fit_transfor does the training and applying, transform only does applying.
since we can't use any info from test, and we need to do the same modification to testing set as well as training set.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''fit -> train
transform -> predict
fit -> calc mean, std for each col
transform -> apply (x-mean)/std to change all to same scale, think it as using training mean on your test data
'''

# 3.2 training and selection
# @title build models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
classifier_RF = RandomForestClassifier()
classifier_KNN = KNeighborsClassifier()
classifier_log = LogisticRegression()
# train model
classifier_log.predict(X_test)
# Accuracy of test data
classifier_log.score(X_test, y_test)

# use 5-fold cross-validation to get the accuracy for different models
model_name = ['logistic', 'KNN', 'randomForest']
model_list = [classifier_log, classifier_KNN, classifier_RF]
count = 0
for classifier in model_name:
    cv_score = model_selection.cross_val_score(classifier, X_train, y_train, cv=5)
    print(cv_score)
    print('Model accuracy of %s is: %.3f'%(model_name[count], cv_score.mean()))
    count += 1

# do prediction with svm model


# 3.3 use Grid search to find optimal hyperparameters
from sklearn.model_selection import GridSearchCV
# helper function for printing out grid search results
def print_grid_search_metrics(gs):
    print("Best score: %0.3f"% gs.best_score_)
    print("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r"%(param_name, best_parameters[param_name]))

# find optimal hyperparameter - logistic regression
# possible hyperparameter options for logistic regression regularization
# penalty is chosen from L1 or L2
# C is the lamda value(weight) for L1 and L2
# ('11',1)('11',5)('11',10)('12',1)('12',5)('12',10)
parameters = {
    'penalty': ('11','12'),
    'C': (1,5,10)
}    
Grid_LR = GridSearchCV(LogisticRegression(), parameters, cv = 5)    
Grid_LR.fit(X_train, y_train)
# the best hyperparameter combination
print_grid_search_metrics(Grid_LR)
# best model
best_LR_model = Grid_LR.best_estimator_

# find optimal hyperparameters: KNN
# possible options, choose K
parameters = {
    'n_neighbors' : [3, 5, 7, 10]
}
Grid_KNN = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)
Grid_KNN.fit(X_train, y_train)
# best K
print_grid_search_metrics(Grid_KNN)

# find optimal hyperparameters: Random Forest
# possible options, choose the number of trees
parameter = {
    'n_estimators': [40,60,80]
}
Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)
# best number of trees
print_grid_search_metrics(Grid_RF)
# best random forest
best_RF_model = Grid_RF.best_estimator_
