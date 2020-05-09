import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import KFold 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sns

def Correlation_Matrix():
    df=pd.read_csv("./EntireFile.csv")
    df["Vac1"]=[0 if x <50 else 1  for x in df["characteristics: vacsIndex"]]
    df.drop("characteristics: vacsIndex",axis=1,inplace=True)
    cor = df.corr()
    cor_target = abs(cor["Vac1"]) #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.3]
    print(relevant_features)
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()

def ifef(col):
    col = str(col)
    if col=="Black":
        return  1
    elif col=="White":
        return 0
    else:
        return 2

def check_svm(x,y):
    from imblearn.over_sampling import SMOTE
    X_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, stratify=y,random_state=123)
    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train) 
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C,max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='rbf', C=C),
              svm.SVC(kernel='rbf', gamma=0.7),
              svm.SVC(kernel='rbf'),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=C), 
              svm.SVC(kernel='linear', C=C, probability=True,random_state=0))
    models = (clf.fit(X_resampled,y_resampled.values.ravel()) for clf in models)
    y_pred = (clf.predict(x_test) for clf in models)
    print([accuracy_score(y_test,y_pred1) for y_pred1 in y_pred])

def read_files():
    x=pd.read_csv("./Features.csv")
    x.dropna(inplace=True)
    x.drop(['molecule','characteristics: sex'],axis=1,inplace=True)
    x["Race"]=x["characteristics: race"].apply(ifef)
    x.drop("characteristics: race",axis=1,inplace=True)
    x=x.astype(float)
    y_1=pd.read_csv("Prediction.csv")
    #print(x.describe())
    y=y_1.drop("Sample name",axis=1)
    y.dropna(inplace=True)
    #print(y.columns)
    y=y.astype(int)
    y["Vac1"]=[0 if x <50 else 1  for x in y["characteristics: vacsIndex"]]
    y.drop("characteristics: vacsIndex",axis=1,inplace=True)
    
    oversample = SMOTE()
    X_over, y_over = oversample.fit_resample(x, y)
    return X_over,y_over,x,y

def SVM_model(X_over,y_over):
    kf = KFold(n_splits=10) # Define the split - into 10 folds 
    C = 1.0
    scores=[]
    precision=[]
    recall=[]
    best_svr =SVC(kernel='linear', C=C)
    best_knn= KNeighborsClassifier(n_neighbors=10)
    for train_index, test_index in kf.split(X_over):
        x_train =X_over.iloc[train_index]
        x_test =X_over.iloc[test_index]
        y_train=y_over.iloc[train_index]
        y_test =y_over.iloc[test_index]
        best_svr.fit(x_train, y_train.values.ravel())
        y_pred=best_svr.predict(x_test)
        scores.append(best_svr.score(x_test, y_test))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))

    print(np.mean(precision))
    print(np.mean(recall))
    print(np.mean(scores))

def KNN_model(no_ne,x_train,x_test,y_test,y_train,X_over,y_over):
    #Create a k-NN classifier with 6 neighbors
    knn = KNeighborsClassifier(n_neighbors=no_ne)
    knn.fit(x_train, y_train.values.ravel()) 
    y_pred = knn.predict(x_test)

    print("Accuracy:",accuracy_score(y_test,y_pred))
    lr_train_score=knn.score(x_train,y_train)
    lr_test_score=knn.score(x_test,y_test)
    print ("KNN training score:", lr_train_score)
    print ("KNN test score: ", lr_test_score)
    #cross validate the results to see if KNN model is the best model or not
    cv_scores = cross_val_score(knn, X_over, y_over.values.ravel(), cv=10)#print each cv score (accuracy) and average them
    print(cv_scores)
    print('cv_scores mean:{}'.format(np.mean(cv_scores)))

def KNN_plot(X_over,y_over):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    # To get a graph to understand which number of neighbours gives us the optimum results
    x_train,x_test,y_train,y_test=train_test_split(X_over,y_over,test_size=0.3,stratify=y_over,random_state=123)
    neighbors = np.arange(1, 40)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
 
    #Loop over different values of k
    for i, k in enumerate(neighbors):
        #Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)
    
        #Fit the classifier to the training data
        knn.fit(x_train,y_train.values.ravel())
         
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(x_train, y_train.values.ravel())
    
        #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(x_test, y_test)

    # Generate The plot of no of neighbours vs accuracy for both testing and training data
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    KNN_model(6,x_train,x_test,y_test,y_train,X_over,y_over)
    
def main():
    X_over,y_over,x,Y=read_files()
    Correlation_Matrix()
    check_svm(x,Y)
    SVM_model(X_over,y_over,x,Y)
    KNN_plot(X_over,y_over)

if __name__=="__main__":
    main()