import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
# part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    # Your code here
    degrees = [1, 3, 6, 9]
    table = np.zeros((4,100))
    
    predict = np.linspace(0.0, 10.0, num=100).reshape((100, 1))
    
    # reshape
    X_F1 = x.reshape((len(x),1))
    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_F1)
        p_poly = poly.fit_transform(predict)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
        linreg = LinearRegression().fit(X_train, y_train)
        table[i] = linreg.predict(p_poly)
    
    # return a numpy array with shape (4, 100)
    return table

# answer_one()

# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

# plot_one(answer_one())

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    
    # Your code here
    degrees = list(range(10))
    
    X_2d = x.reshape(len(x), 1)
    r2_train = np.zeros((10,))
    r2_test = np.zeros((10,))
    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_2d)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
        linreg = LinearRegression().fit(X_train, y_train)
        r2_train[i] = r2_score(y_train, linreg.predict(X_train))
        r2_test[i] = r2_score(y_test, linreg.predict(X_test))
            
    return (r2_train, r2_test)

# answer_two()

def answer_three():
    
    # Your code here
    
    return (0,9,7)
	
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    from sklearn.linear_model import Lasso
    # from sklearn.preprocessing import MinMaxScaler
    
    # Your code here
    # scaler = MinMaxScaler()
    
    X_2d = x.reshape(len(x),1)
    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, random_state=0)
    
    # linreg
    poly = PolynomialFeatures(degree=12)
    X_poly = poly.fit_transform(X_2d)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
    linreg = LinearRegression().fit(X_train, y_train)
    LinearRegression_R2_test_score = r2_score(y_test, linreg.predict(X_test))

    # Lasso Regression model(alpha=0.01, max_iter=10000)
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    linlasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
    Lasso_R2_test_score = r2_score(y_test, linlasso.predict(X_test))
    
    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

# answer_four()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # Your code here
    dtc = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    importance = pd.DataFrame(X_train2.columns)
    importance = importance.merge(pd.DataFrame(dtc.feature_importances_), left_index=True, right_index=True)
    
    # top 5 important features
    top5 = list(importance.sort_values(by='0_y', ascending=False).head(5)['0_x'])
    return top5

# answer_five()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here
    # Use X_subset and y_subset
    param_range = np.logspace(-4,1,6)
    # print(param_range)
    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                            param_name='gamma',
                                            param_range=param_range)
    return (train_scores.mean(axis=1), test_scores.mean(axis=1))

# answer_six()

def answer_seven():
    
    # Your code here
    
    return (0.0001, 10, 0.1)