
"""
============Imports================
"""

# Importing necessary models.

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import tree

# Measuring time elapsed

from time import perf_counter

# Accessing attributes of Decision Tree Classifier

from inspect import getmembers

# Rescaling the original MINST images

from PIL.Image import fromarray

# Spiltting data

from sklearn.model_selection import train_test_split

# Finding the best parameters for models

from sklearn.model_selection import GridSearchCV

# Loading MINST dataset from local files

from mlxtend.data import loadlocal_mnist

# Managing data

import numpy

# Clearing the console

from os import system

# Plotting graphs

import matplotlib.pyplot as plt

# Removing the toolbar from graphs

from matplotlib import rcParams
rcParams['toolbar'] = 'None'


"""
============Functions================
"""


# Train Decision Tree Model on the dataset,
# then show an image depicting feature importances.
# This creates visual representation of which pixel
# is the most important.

def print_tree_importances(X,y):
    mlp = tree.DecisionTreeClassifier(max_depth=50, criterion='entropy')
    mlp.fit(X,y)
    for i in getmembers(mlp):
        if i[0] == 'feature_importances_':
            plt.matshow(i[1].reshape(28, 28), cmap=plt.cm.gray)
            plt.show()

# Train Multilayer Perceptron Model on the dataset,
# then show a series of images depicting node coefficients
# of the inner hidden layer. Print out coefficients of the
# output layer to visualise which nodes affect them the most.

def visualise_mlp(n_numbers, layers, plots1, plots2):
    mlp = MLPClassifier(alpha=0.00001, max_iter=200, hidden_layer_sizes=(layers,), solver='sgd')
    mlp.fit(X_train[y_train < n_numbers], y_train[y_train < n_numbers])
    fig, axes = plt.subplots(plots1, plots2, figsize=(5, 4))
    for i, ax in enumerate(axes.ravel()):
        coef = mlp.coefs_[0][:, i]
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(i + 1)
    for i in range(n_numbers):
        print('Number :', i)
        temp = []
        for q in range(layers):
            temp.append([mlp.coefs_[1][:, i][q], q])
        for p in sorted(temp, key=lambda v: v[0]):
            print(p[0], 'for node nr', p[1] + 1)
    plt.show()


# Getting the MINST dataset, and rescaling the
# original image to the selected edge size.
# Returning the data split into training and test set.

def get_data(a, rand):
    print('Size of frames =',a,'x',a)
    train_data_x, train_data_y = loadlocal_mnist('td.idx3-ubyte', 'tl.idx1-ubyte')
    test_data_x, test_data_y = loadlocal_mnist('trd.idx3-ubyte', 'trl.idx1-ubyte')
    data_x=numpy.concatenate((train_data_x,test_data_x))
    data_y=numpy.concatenate((train_data_y,test_data_y))
    X = numpy.array([numpy.array(fromarray(x.reshape(28, 28)).resize((a, a))).reshape(a * a) for x in data_x])
    y = data_y
    if rand==-1:
        return train_test_split(X, y)
    else:
        return train_test_split(X, y, random_state=rand)


# Testing different parameters on models and printing out the results.

def do_grid_search(mlp, parameter_space, X, y):
    number_of_models = 1
    for i in parameter_space:
        number_of_models *= len(parameter_space.get(i))
    t = perf_counter()
    mlp.fit(X, y)
    print('This test will take', round((perf_counter() - t) * number_of_models, 2), 's')
    t = perf_counter()
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=10)
    clf.fit(X, y)
    print('This took', round(perf_counter() - t, 3), 's')
    print('Best parameters found:\n', clf.best_params_)
    t = perf_counter()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    temp = []
    for i in range(len(means)):
        temp.append([means[i], stds[i], clf.cv_results_['params'][i]])
    for i in sorted(temp, key=lambda x: x[0]):
        print("%0.3f (+/-%0.03f) for %r" % (i[0], i[1] * 2, i[2]))
    print('And this', round(perf_counter() - t, 3), 's')
    

# Scoring available models, 
# returning the results and the model with the highest accuracy

def get_best_models(X_train, y_train, X_test, y_test):
    names = ['KNN',
             'Centroids',
             'Decision Tree',
             'MLP',
             'Gaussian Naive-Bayes']
    models = [KNeighborsClassifier(n_neighbors=4, weights='distance', algorithm='brute', metric='cosine'),
              NearestCentroid(),
              tree.DecisionTreeClassifier(max_depth=10, criterion='entropy'),
              MLPClassifier(activation='logistic', alpha=0.00001, max_iter=1000),
              GaussianNB(var_smoothing=0.1)]
    accs = []
    for i in range(len(models)):
        print('Testing Model:',names[i])
        mod = models[i].fit(X_train, y_train)
        acc = models[i].score(X_test, y_test)
        accs.append([acc, mod])
    try:
        best = max(accs, key=lambda x: x[0])
    except:
        print(accs)
    mlp = best[1]
    return mlp, accs


# Predicting the results of the test set, and printing out
# consecutively 10 correct or incorrect predictions depending
# on the choice.

def someloop(X_test, y_test, mlp, a, mode=0):
    y_pred = mlp.predict(X_test)
    incorrect = X_test[y_pred != y_test]
    incorrect_true = y_test[y_pred != y_test]
    incorrect_pred = y_pred[y_pred != y_test]
    count = 0
    if mode==0:
        for cyfra in X_test:
            print("\ntrue value:", y_test[count])
            print("predicted value:", y_pred[count])
            plt.matshow(cyfra.reshape(a, a), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
            plt.show()
            count += 1
            if count==10:
                break
    else:
        for cyfra in incorrect:
            print("\ntrue value:", incorrect_true[count])
            print("predicted value:", incorrect_pred[count])
            plt.matshow(cyfra.reshape(a, a), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
            plt.show()
            count += 1
            if count==10:
                break
            


# Creating the graph of all model performances
# depending on the size of the image.


def attribute_graph():
    fig, ax = plt.subplots()
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=28)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 1.1)
    ax.grid()
    labels = ['KNN',
              'Centroids',
              'Decision Tree',
              'MLP',
              'Gaussian Naive-Bayes']
    data = []
    for p in range(28):
        data.append([])
        for q in range(len(labels)):
            data[p].append(0)
    for d in range(1):
        for i in range(28):
            X_train, X_test, y_train, y_test = get_data(28 - i, d)
            mlp, accs = get_best_models(X_train, y_train, X_test, y_test)
            for y in range(len(accs)):
                data[i][y] += accs[y][0] / 1
    plt.plot(range(28)[::-1], data)
    plt.legend(labels)
    plt.show()


"""
============Interface================
"""


if __name__ == "__main__":
    while True:
        try:
            defaults=int(input('Use defaults :'))
            if defaults not in [0,1]:
                print('Wrong input. Available options for default:\n    0 - user settings\n    1 - default settings')
            else:
                break
        except:
            print('Wrong input. An integer was expected.')
    while True:
        if defaults==0:
            try:
                pix = int(input('MINST image size :'))
                if pix<1 or pix>28:
                    print('Wrong input. Image size should be within range (1,28)')
                else:
                    break
            except:
                print('Wrong input. An integer was expected.')
    while True:
        try:
            randstate = int(input('Random state for Data Split, enter -1 for random :'))
            if randstate<-1 or randstate>2**32:
                print('Wrong input. Image size should be within range (1,2**32)')
            else:
                break
        except:
            print('Wrong input. An integer was expected.') 
    else:
        pix=28
        randstate=-1
    X_train, X_test, y_train, y_test = get_data(pix, randstate)
    while True:
        system('cls')
        print("""Choose an option:
        0 - Predict MINST Dataset and iterate over the results.
        1 - Visualise Pixel Importances in Decision Tree Model.
        2 - Visualise Multi Layer Perceptron Node Coefficients.
        3 - Show graph of all model performances against image size.
        q - Quit the programme.""")
        option=input(':')
        if option not in ['0','1','2','3','q']:
            print('Wrong input.')
        else:
            system('cls')
            if option=='0':
                if defaults==0:
                    try:
                        mode=int(input('Select the mode:\n  0 - correct predictions\n  1 - incorrect predictions\n:'))
                        if mode in [0,1]:
                            mlp=get_best_models(X_train, y_train, X_test, y_test)[0]
                            someloop(X_test, y_test, mlp, pix, mode)
                        else:
                            print('Wrong input. There is no such option.')
                    except:
                        print('Wrong input. An integer was expected.')
                else:
                    mode=0
                    mlp=get_best_models(X_train, y_train, X_test, y_test)[0]
                    someloop(X_test, y_test, mlp, pix, mode)
            if option=='1':
                print_tree_importances(X_train,y_train)
            if option=='2':
                if defaults==0:
                    try:
                        digits=int(input('Enter number of digits included :'))
                        if 11>digits>2:
                            nodes=int(input('Enter number of nodes included :'))
                            if 500>nodes>1:
                                v=int(input('Enter number of graph rows :'))
                                h=int(input('Enter number of graph columns :'))
                                if v*h==nodes:
                                    visualise_mlp(digits,nodes,v,h)
                                else:
                                    print('rows x columns should equal the number of nodes')
                            else:
                                print('Nodes should be within range (1,500)')
                        else:
                             print('Digits should be within range (2,10)') 
                        
                    except:
                        print('Wrong input. An integer was expected.')
                else:
                    digits=4
                    nodes=6
                    v=2
                    h=3
                    visualise_mlp(digits,nodes,v,h)
            if option=='3':
                attribute_graph()
            if option=='q':
                break
        


