# Import scikit-learn dataset library
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Loading the cancer dataset from sklearn
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Splitting the data into (80-20%) train-test split using sklearn.model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5-fold cross validation sklearn.model_selection
# Using GridSearchCV for hyperparameter tuning
# Modelling a RandomForestClassifier from sklearn.ensemble
# Using the metric accuracy, precision, recall and f1 to evaluate model performance during 5-fold cross validation
# Prioritizing recall (sensitivity) for medical use
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    {   
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': range(1, 10),
        'criterion':['gini','entropy']
    }, 
    cv=kf, 
    scoring='recall',
    n_jobs=-1
)

# Training the model
grid_search.fit(X_train, y_train)

# Getting the optimal trained model and printing best model and hyperparameters
best_model = grid_search.best_estimator_
print("Best Score:", grid_search.best_score_)
print("Best Hyperparameters:", grid_search.best_params_)

# Extracting the  predictive values from model for test set 
y_pred = best_model.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("======================================================")
print("=================== Confusion Matrix =================")
print("======================================================")
print(conf_matrix)
tn, fp, fn, tp = conf_matrix.ravel()

print("======================================================")
print("===================== Model Scores ===================")
print("======================================================")
print(metrics.classification_report(y_test, y_pred))
print("Specificity :", tn / (tn + fp))
print("F-1 Score   :", metrics.f1_score(y_test, y_pred))