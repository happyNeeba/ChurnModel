# the correlated features that we are interested in:
# (total_day_minutes, total_day_charge),(total_eve_minutes, total_eve_charge),(total_intl_minutes, total_intl_charge)
# add L1 regularization to logistic regression, check the coef for feature selection
scaler = StandardScaler()
X_l1 = scaler.fit_transform(X)
LRmodel_l1 = LogisticRegression(penalty = "l1", C = 0.1)
LRmodel_l1.fit(X_l1, y)
LRmodel_l1.coef_[0]
print ("Logistic Regresssion Coefficients")
for k,v in sorted(
        zip(
            map(lambda x: round (x, 4), LRmodel_l1.coef_[0]), 
        churn_feat_space.columns), key = lambda k_v: (-abs(k_v[0], k_v[1])
    )
):
    print(v + ": " + str(k))
# add L2 regularization to logistic regression
# check the coef for feature selection
scaler = StandardScaler()
X_l2 = scaler.fit_transform(X)
LRmodel_l2 = LogisticRegression(penalty = "l2", C = 0.1)
LRmodel_l2.fit(X_l2, y)
LRmodel_l2.coef_[0]
print("Logistic Regresssion (L2) Coefficients")
for k,v in sorted(
    zip(
        map(lambda x: round(x, 4), LRmodel_l2.coef_[0]),
        churn_feat_space.columns), key= lambda k_v :(-abs(k_v[0]), k_v[1])
    )
):
    print(v + ": " + str(k))

# RF feature importance discussion
# check feature importance of RF
forest = RandomForestClassifier()
forest.fit(X, y)
importances = forest.feature_importances_
# print feature ranking
print("Feature importance ranking by RF Model:")
for k,v in sorted(
    zip(
        map(lambda x: round(x, 4), importances), churn_feat_space.columns
    ),reverse = True
):
    print(v + ": " + str(k))