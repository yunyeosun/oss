#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/yunyeosun/oss

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sklearn.metrics as mt

def load_dataset(dataset_path):
	#To-Do: Implement this function
	df = pd.read_csv(dataset_path) 
	return df

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	n_feats = dataset_df.drop(columns="target", axis=1).shape[1]
	n_class1=len(dataset_df.loc[dataset_df['target'] == 1]) #data_df.groupby("target").size
	n_class0=len(dataset_df.loc[dataset_df['target'] == 0])
	return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x=dataset_df.drop(columns="target", axis=1)
	y=dataset_df["target"]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testset_size)
	return x_train, x_test, y_train, y_test
	
def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)	
	acc = mt.accuracy_score(y_test, dt_cls.predict(x_test))
	prec =mt.precision_score(y_test, dt_cls.predict(x_test))
	recall = mt.recall_score(y_test, dt_cls.predict(x_test))
	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rt_cls = RandomForestClassifier()
	rt_cls.fit(x_train, y_train)	
	acc = mt.accuracy_score(y_test, rt_cls.predict(x_test))
	prec =mt.precision_score(y_test, rt_cls.predict(x_test))
	recall = mt.recall_score(y_test, rt_cls.predict(x_test))
	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_cls = SVC()
	svm_cls.fit(x_train, y_train)
	svm_pipe = make_pipeline(StandardScaler(), SVC())
	svm_pipe.fit(x_train, y_train)
	acc = mt.accuracy_score(y_test, svm_pipe.predict(x_test))
	prec = mt.precision_score(y_test, svm_pipe.predict(x_test))
	recall = mt.recall_score(y_test, svm_pipe.predict(x_test))
	return acc, prec, recall
	
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