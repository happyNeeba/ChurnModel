# 3.4 Model Evaluation 
# 3.4.1 confustion matrix (precision, recall, accuracy)
'''class of interest as positive
 TP: correctly labeled real churn
 Precision (PPV, positive predictive value): tp/(tp + fp);
 total number of true predictive churn divided by the total number of predictive churn;
 high precision means low fp, not many return users were predicted as churn users
 Recall(sensitivity, hit rate, true positive rate): tp/(tp + fn)
 predict most positive or churn users correctly
 high recall means low fn, not many churn users were predicted as return users
'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# calculate accuracy, precision, recall, [[tn, fp],[]]
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print(classifier)
    print("Accuracy is: %0.3f" % accuracy)
    print("precision is: %0.3f"% precision)
    print("recall is: %0.3f"% recall)
# print out confusion matrices
def draw_confusion_matrices(cms):
    class_names = ['Not', 'Churn']
    for cm in cms:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)
        fig = plt.figure()
        ax = fig.add_subplot(111) # what is 111?
        cax = ax.matshow(cm, interpolation = 'nearest', cmap=plt.get_cmap('Reds'))
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
%matplotlib inline
# confusion matrix, accuracy, precision and recall for random forest and logistic regression
confusion_matrices = [
    ('RF', confusion_matrix(y_test, best_RF_model.predict(X_test))),
    ('LR', confusion_matrix(y_test, best_LR_model.predict(X_test)))
]
draw_confusion_matrices(confusion_matrices)

# 3.4.2 ROC & AUC
# ROC of RF model
from sklearn.metrics import roc_curve
from sklearn import metrics
# Use predict_proba to get the probability results of RF
y_pred_rf = best_RF_model.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
# ROC curve of RF result
plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_rf, tpr_rf, label = 'RF')
plt.xlabel('False positive rate')
plt.ylable('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc = 'best')
plt.show()

from sklearn import metrics

# AUC score
metrics.auc(fpr_rf, tpr_rf)

# ROC of LR model
y_pred_lr = best_LR_model.predict_proba(X_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
# ROC curve
plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_rf, tpr_rf, label = 'LR')
plt.xlabel('False positive rate')
plt.ylable('True positive rate')
plt.title('ROC curve - LR model')
plt.legend(loc = 'best')
plt.show()
# AUC score
metrics.auc(fpr_rf, tpr_rf)