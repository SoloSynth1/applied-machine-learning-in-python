import numpy as np
import pandas as pd


def answer_one():
    
    # Your code here
    df = pd.read_csv('fraud_data.csv')
    return df['Class'].mean()

# answer_one()

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score
    
    # Your code here
    dc = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    y_dc = dc.predict(X_test)

    a_score = accuracy_score(y_test, y_dc)
    r_score = recall_score(y_test, y_dc)
    return (a_score, r_score)

# answer_two()

def answer_three():
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.svm import SVC

    # Your code here
    svc = SVC().fit(X_train, y_train)
    y_svc = svc.predict(X_test)
    
    a_score = accuracy_score(y_test, y_svc)
    r_score = recall_score(y_test, y_svc)
    p_score = precision_score(y_test, y_svc)
    return (a_score, r_score, p_score)

# answer_three()

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here
    params =  {'C': 1e9, 'gamma': 1e-07}
    svc = SVC().set_params(**params).fit(X_train, y_train)
    y_svc = svc.decision_function(X_test)
    y_svc[y_svc > -220] = 1
    y_svc[y_svc != 1] = 0
    c_matrix = confusion_matrix(y_test, y_svc)
    return c_matrix

# answer_four()

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
#     import matplotlib.pyplot as plt
    
    # Your code here
    logreg = LogisticRegression().fit(X_train, y_train)
    y_score_lr = logreg.predict_proba(X_test)
    p, r, t = precision_recall_curve(y_test, y_score_lr[:,1])
    tprate, fprate, _ = roc_curve(y_test, y_score_lr[:,1])
#     plt.plot(tprate, fprate, label='ROC')
#     plt.plot(p, r, label='Pre-Rec C')
#     plt.show()
    
    recall_queried = r[ np.argmin(np.abs(p - 0.75))]
    tprate_queried = tprate[ np.argmin(np.abs(fprate - 0.16))]
    return (recall_queried, tprate_queried)

# answer_five()

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Your code here
    lr = LogisticRegression()
    params = {'C':[0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    gs = GridSearchCV(lr, param_grid=params, scoring='recall').fit(X_train, y_train)

    result = np.array(gs.cv_results_['mean_test_score']).reshape((5,2))
    return result

# answer_six()

# # Use the following function to help visualize results from the grid search
# def GridSearch_Heatmap(scores):
#     %matplotlib notebook
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     plt.figure()
#     sns.heatmap(scores.reshape((5,2)), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
#     plt.yticks(rotation=0);
    
# # GridSearch_Heatmap(answer_six())