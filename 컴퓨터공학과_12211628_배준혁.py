#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/mearae/opensorce_SW_instroduction.git

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_dataset(dataset_path):
	#To-Do: Implement this function
    return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
    n_cls0, n_cls1 = dataset_df.groupby("target").size()
    return len(dataset_df.columns)-1, n_cls0, n_cls1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
    X = dataset_df.drop(columns="target", axis=1)
    y = dataset_df["target"]
    return train_test_split(X, y, test_size = testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    dt_cls = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier()
    )
    dt_cls.fit(x_train, y_train)
    pred_x_dt = dt_cls.predict(x_test)
    return accuracy_score(y_test, pred_x_dt), precision_score(y_test, pred_x_dt), recall_score(y_test, pred_x_dt)
            
def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    rf_cls = make_pipeline(
        StandardScaler(),
        RandomForestClassifier()
    )
    rf_cls.fit(x_train, y_train)
    pred_x_rf = rf_cls.predict(x_test)
    return accuracy_score(y_test, pred_x_rf), precision_score(y_test, pred_x_rf), recall_score(y_test, pred_x_rf)

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    svm_cls = make_pipeline(
        StandardScaler(),
        SVC()
    )
    svm_cls.fit(x_train, y_train)
    pred_x_svm = svm_cls.predict(x_test)
    return accuracy_score(y_test, pred_x_svm), precision_score(y_test, pred_x_svm), recall_score(y_test, pred_x_svm)

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)