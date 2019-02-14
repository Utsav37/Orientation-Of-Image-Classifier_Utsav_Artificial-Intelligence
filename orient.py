#!/usr/bin/env python3
###########################################################################  RANDOM FOREST  ################################################################################################
# Random Forest implemented by me achieved 70.41 % Accuracy.

# you need to pass ./orient.py test test_file.txt forest_model.txt forest     in order to run Random Forest

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# IMP NOTE: 				IMP NOTE:				 IMP NOTE:				 IMP NOTE:				 IMP NOTE:				 	IMP NOTE:				 IMP NOTE:   	  		IMP NOTE:  
# My model achieved 70.41% accuracy for Random Forest. I saved that file in forest_model.txt. If you test the code, you will definitely get 70% accuracy. 
# It took 3-5 hours for me to train my model.
# BUT, only and only for AI's ease of checking my code, I replaced one line with another which works under 1 minutes but gives accuracy of nearly 63.5% to 67.5%.
# IF AI's want to train model with  70.41% accuracy, they will have to replace one line with another which is mentioned below.

# Thus: Random Forest:   for testing:    forest_model.txt ---> 70.41% Accuracy ----> forest_model.txt was achieved by 3 hours training
#                        for training :  For ease of AI's to check whether Code is Working or not : 64.5% to 67.2% accuracy----> will take 3 minutes hardly to run
                       
#                        IF AI's want model with 70.41% Accuracy in training also(Though after knowing it will take 3-4 hours or more time ), 
#                        then please comment line-----> for c in [128]:    
#     				   and uncomment line --------->  # for c in range(1,255):
#     				   INSIDE RANDOM FOREST CODE (Line 446 and Line 447)
                       
#                        1st line was just splitting on 128 for each column(Thus, it achieved just 66% avg accuracy), 
#                        while second line would split on 1 to 255 values for each column.(Thus it achieved 70.41% accuracy)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Over all in the model We were behind tuning 5 parameters: 
# 					1. level or depth of Decision Trees
# 					2. n_trees : number of trees to explore for Random Forest
# 					3. n_cols : number of columns to take in each Decision tree.
# 					4. n_rows: number of rows to take in each Decision Tree.
# 					5. value_of_split: (MOST IMPORTANT)   The value at which we need to split for each column for checking.
# Approach 1 : We tried to implement by splitting on 128 for each column and after lot of parameter tuning, we achieved accuracy of nearly 68%-70%. I ran 10,000 loops 
# 			and got 5 parameters that got this much high accuracy, but while checking for same parameters in same program with no parameter tuning loops, 
# 			we dropped accuracy from 69-71% to directly 65-67%. 
# Approach 2: We tried to split values based on Median. We created medians for each column and whenever we need to split on particular column, we would split on that median for that
# 			particular column. This approach was also nearly 65% to 68%. 
# Approach 3: We tried to split on [25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205,215,225] values for each column. 
# 			So whenever we will go for finding best split for a particular set of rows, we will loop through all columns and for all columns, we will split on all above mentioned values
# 			, and surprisingly we achieved accuracy of 66%-69%. Thus, this idea worked!!! 
# Approach 4: Now, we thought that it would be great if we could find best point splitting such that that point is best value of split among all possible splits.
# 			So, idea was to use split values of 1 to 255 for each column and to do so for all 192 columns , and then to come up with that golden column and that diamond splitting value for that column.
# 			That value was the most accurate split among all possible ways. BUT< here I had to do training time of 3-5 hours and then I got my final answer of forest_model.txt.


# JUST FOR AI's EASY CODE CHECKING: I wrote Random Forest code (for train portion) which by default trains a model with 65-68% accuracy which is splitting only on 128.
# 	I made  program such that if AI's want my 70.41% model then>>>comment one line--> for c in [128]:  and uncomment this line : # for c in range(1,255):  
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     PROGRAM EXPLANATION   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Program Explanation of Random Forest: Random Forest consists of Decision Trees. The program simply consists of basic building block functions which does their own job.
# The name of the functions itself will tell what they do.
# 1. def label_count(rows): This function counts the number of rows that are of certain category and returns dictionary ex: 0 : 3005 90: 4332 180: 4833 270 : 3555
# 2. def most_common(list): This function counts the most common element from a list and  returns that element. This is used to count most common element when we are terminating at leaf node.
						  # Example: If at leaf node we want to terminate, we will have to assign one orientation, so we will assign 0 90 180 or 270 and return that as the most common element.
# 3. def entropy(rows): Given a set of say 4000 rows, it will calculate entropy or disorderness among this 4000 rows. For this, it will have to take into account all label's probability of coming.
# 4. def info_gain(left, right, current_uncertainty): This function will get left child rows and right child rows (obtained by say : column  value <128, put that row in left child set of rows and 
														# say if >128 than put that row in right child set of rows) and current uncertainty that will be of parent of this nodes. 
														# Function calculates information gain by taking into account parents entropy and left and right childs entropy.
# 5. def partition(rows,colindex, valueofsplit): This function just does partitions based on the column index given and for that column, the split value given. SO we will compare that whole column 
												# and which ever rows are greater than value of split , they will enter in true_rows and rest rows in false_rows. Thus, it returns true_rows and false_rows.

# 6. def best_split(rows): It counts the best gain possible for set of rows by splitting and column values for which that gain could be achieved and split values for that column which is used.
							# Thus it returns best_gain, best_col, and best_value_of_split
# 7. def decision_tree(rows,level,randomcols):This function creates decision trees by using all above functions and input given here is dataset that is sliced(26000) 
							# and level uptill which that decision tree can grow. This function outputs a dictionary and that dictionary is our single decision tree itself.
# 8. def random_forest(trainrgbvalues,n_trees,trainorientation,level,n_cols,n_rows): It creates number of decision trees as mentioned in n_trees as argument and uses values passed to it ie.
						# n_rows - no of rows per decision tree , n_cols- no of columns per decision tree, level - level upto which one decision tree can be explored. 
						# trainrgbvalues- list that contains all rows which are again in list (all training data is passed here ) and trainorientation - the 0 90 180 270 of each row is also passed here 
# 9. def func(tree, testlist): This function takes two arguments: tree- dictionary form of tree and testlist - row for testing. Function will use the tree to explore decision tree and give output orientation.
						# This function will be called for each and every tree that are in  a dictionary and this all for each row and at the end if we are having 20 trees in our final model,
						# we will have  20 orientations for each row and that row will be assigned orientation that is repeated the most .
#########################################################################################################################################################################################
# REFERENCES: From Google Developers:
# 					https://www.youtube.com/watch?v=LDRbO9a6XPU
# 					https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb

###############################################################################################################################################################################

"""
KNN:
Implementation:
1. Created a class for the KNN model
2. The class is initialized with 2 things a k value and the distance function to be used.
3. We store this k value which is chosen to be 11 as it gives optimal accuracy which can be
seen from the graph below of k values vs accuracy and the distance metric used in
model.txt file during the training phase.
4. KNN is a lazy algorithm so this is the most we can save as a KNN model.
5. During the testing phase for each testing sample the distances from the testing sample
to all other training samples are found out and using those the nearest k neighbors are
found out and the mode of the neighbors is given as the prediction for the testing sample.
Functions used:
def __init__ - Function to initialize the KNN model.
def distance_l2 - Function calculates the Euclidean distance between given 2 samples using the
192 features.
def predict - Function that generates the predictions for each testing sample by taking the
mode of the k closest neighbors.

Patterns observed :
1. Almost all of the landscape images are classified successfully, this must be because of the
clear distinction of light blue pixels for the sky and the darker pixels for the landscape.
2. Images where such a contrast is not observed are miss classified by our model.

ADABOOST model
Implementation:
In the AdaBoost model we use the weak classifier as a decision stump.
Train and test data pre-processing:
* Of the total of 192 features we generate 300 new features.
* We generate the combinations of all features in pairs of two.
* Therefore we have 192x192 number of combinations of pairs.
* Of these we randomly choose 300 pairs
* And we now generate features by subtracting the first feature number from the second.
* We now add a weights column required for boosting whose initial value
Model implementation:
1. As mentioned above we use decision stumps as weak classifier.
2. We have 2 classes:
a) DecisionTree
b) AdaBoost
3. The decision tree class is initialized by giving it the label we are interested in and the
feature column to use for the stump.
4. The decision tree then scans for values from -200 to 200 with a step size of 20 as split
values, generates temporary splits, calculates the entropies, and the information gain of
each split value for the given feature.
5. We use shannon’n formula for entropy calculation.
Entropy    p log( p)
6. We then choose the split value having t
he max information gain and return this split to the
adaboost model.
7. The adaboost model then does the split, calculates the error and then updates the weights
of the correctly classified samples using the formula below:
weight = weight * error /(1 - error)
8. We then normalize the weights.
9. The model then calculates the score of the weak classifier as
score = log(1 - error / error)
10. This is done till there are 100 weak classifiers accumulated.
One vs All approach:
We have used a one vs all approach here, so we have 4 models one for 0, one for 90, one for
180 and one for 270. We get the predictions from the 4 models of the positive(0,90,180,270)
classes which are in the form of values. We predict a testing sample to be of the class for which
the prediction has the highest place.


"""

import numpy as np
import time
import sys
import ast

import pdb
import pandas as pd
from collections import defaultdict

import operator
from numpy import log2 as log
eps = np.finfo(float).eps
import math
import pickle
import random
import itertools

# Function to calculate accuracy
def score(predictions,actual):
	error_count = 0
	for entry in range(len(predictions)):
		if predictions[entry] != actual[entry]:
			error_count += 1
	return (1-error_count/len(predictions))*100

# KNN implementation
class KNN:
	def __init__(self,k,distance_func):
		self.k = k
		self.distance_func = distance_func
    
	def distance_l2(self,train_num,test_num):
		train_feature_array = self.train_data[train_num,1:]
		test_feature_array = self.test_data[test_num,1:]
		distance = np.sum(np.square(train_feature_array - test_feature_array))
		return distance
    
	def predict(self,train_data,test_data):
		self.train_data = train_data
		self.test_data = test_data    
		predictions = []
		
		# For each test sample
		for i in range(self.test_data.shape[0]):			
			# Dictionary to store distances
			d = {}
			
			# Calculate distance from test sample to each train sample
			for j in range(self.train_data.shape[0]):
				d[self.distance_l2(j,i)] = j
				
			# List to store result
			result = []
			# Choose nearest k entries
			for k_temp in range(self.k):
				result.append(self.train_data[d[sorted(d)[k_temp]],0])
            
			# Append mode of k neighbours to predictions
			predictions.extend([max(set(result), key=result.count)])
		
		# Return predictions on test data
		return predictions

# Class for the Decision Tree
class DecisionTree:
	def __init__(self,interest_label,features):
		"""Initialize"""
		self.interest_label = interest_label
		self.features = features
	
	def testSplit(self,feature_col,value):
		"""
		Given a feature and the split value, make the split and return the info gain.
		"""
		# Creating a temporary split
		split_le = []
		split_g = []
		
		split_le = self.df.loc[self.df[feature_col]<=value,['y','w']]
		split_g = self.df.loc[self.df[feature_col]>value,['y','w']]
		
		# Entropy of parent
		p_entropy = self.calculateEntropy(self.df.loc[:,['y','w']],len(self.df))
		# Entrop of left child of split
		l_entropy = self.calculateEntropy(split_le,len(self.df))
		# Entrop of right child of split
		g_entropy = self.calculateEntropy(split_g,len(self.df))
		# Calculate IG
		IG = p_entropy - l_entropy - g_entropy
		
		# print("Parent entropy = ",p_entropy)
		# print("***********************")
		# print("Split 1 entropy = ",l_entropy)
		# print("***********************")
		# print("Split 2 entropy = ",g_entropy)
		# print("***********************")
		# print("Information gain = ",IG)
		
		# le_mode = max(set(split_le), key=split_le.count)
		# g_mode = max(set(split_g), key=split_g.count)
		
		return IG
		
	def calculateEntropy(self,split,parent_total):
		"""
		Calculate entropy of a split
		"""
		total_w = sum(split['w'])
		
		if total_w == 0:
			return 1
			
		# Count of positive examples by weight of the samples
		count_pos = sum(split.loc[split['y'] == self.interest_label,'w']) / total_w
		count_neg = sum(split.loc[split['y'] != self.interest_label,'w']) / total_w
		
		# Total elements in the split
		total = len(split['y'])
		
		# Calculate entropy
		if count_pos!=0:
			entropy = -1 * count_pos * log(count_pos)
		else:
			entropy = 1
		if count_neg!=0:
			entropy += -1 * count_neg * log(count_neg)
		
		return (entropy * total/parent_total)
	
	def train(self,df):
		self.df = df
		
		# List to store information gains for diffrent features and splits
		IG_list = []
		
		for feature_num in self.features:
			for split_val in range(-240,240,20):
				IG = self.testSplit(feature_num,split_val)
				#print("IG for feature",feature_num," and value",split_val," =",IG)
				IG_list.append((IG,feature_num,split_val))
		
		# Max info gain observed
		# print("\n\nMax IG pbserved for - ",max(IG_list, key=operator.itemgetter(0)))
		return max(IG_list, key=operator.itemgetter(0))
		
class AdaBoost:
	def __init__(self,interest_label,n_estimators):
		self.interest_label = interest_label
		self.n_estimators = n_estimators
	
	def train(self,df):
		learners = []
		i = 0
		#features = [x for x in range(1,190)]
		features = 1
		
		while len(learners)<=self.n_estimators:
			# Train the weak learner
			model = DecisionTree(interest_label = self.interest_label, features = [features])
			(IG,feature_col,value) = model.train(df)
			
			print(i+1,"weak learner -",(IG,feature_col,value))
			i += 1
			
			#features.remove(feature_col)
			features += 1 
			
			# Split using the new weak classifier
			
			split_le = df.loc[df[feature_col]<=value,['y','w']]		
			split_g = df.loc[df[feature_col]>value,['y','w']]
	
			# Find predictions
			#le_mode = split_le['y'].mode()[0]
			#g_mode = split_g['y'].mode()[0]
			
			if not split_le.empty:
				le_mode = split_le['y'].value_counts()[0]/len(split_le)
			else:
				continue
			if not split_g.empty:
				g_mode = split_g['y'].value_counts()[0]/len(split_g)
			else:
				continue
			
			if le_mode > g_mode:
				le_mode = self.interest_label
			else:
				g_mode = self.interest_label
			
			
			# Find count of missclassfied samples
			if le_mode == self.interest_label:
				error_le = sum(split_le.loc[split_le['y'] != le_mode,'w'])
				error_g = sum(split_g.loc[split_g['y'] == le_mode,'w'])
				# Get labels of correctly classified samples
				le_correct_index = split_le.loc[split_le['y'] == le_mode].index
				ge_correct_index = split_g.loc[split_g['y'] != le_mode].index
			elif g_mode == self.interest_label:
				error_le = sum(split_le.loc[split_le['y'] == g_mode,'w'])
				error_g = sum(split_g.loc[split_g['y'] != g_mode,'w'])
				# Get labels of correctly classified samples
				le_correct_index = split_le.loc[split_le['y'] != g_mode].index
				ge_correct_index = split_g.loc[split_g['y'] == g_mode].index
			else:
				continue
			
			total_error = (round(error_le,3) + round(error_g,3))/ sum(df['w'])
			
			# Update the weights of correctly classifed samples
			df.loc[le_correct_index,'w'] *= (total_error/(1-total_error))
			df.loc[ge_correct_index,'w'] *= (total_error/(1-total_error))
			
			# Normalize the weights
			df['w'] = df['w'] / sum(df['w'])
			
			# Store the model along with its score
			score = math.log((1-total_error)/total_error)
			if score > 0:
				learners.append((feature_col,value,le_mode,g_mode,score))
			print(score)
			
		return learners
		
	def predict(self,learners,df):
		# Predict the labels
		predictions = []
		for j in range(df.shape[0]):
			prediction = 0
			sample = df.iloc[j]
			for i in range(len(learners)):
				feature_col = learners[i][0]
				value = learners[i][1]
				le_label = learners[i][2]
				g_label = learners[i][3]
				score = learners[i][4]
				
				if le_label == self.interest_label:
					if sample[feature_col] <= value:
						prediction += 1 * score
					else:
						prediction += -1 * score
				else:
					if sample[feature_col] <= value:
						prediction += -1 * score
					else:
						prediction += 1 * score
						
			if prediction > 0:
				predictions.append(prediction)
			else:
				predictions.append(-1)
		return predictions
		
		
todo,todofile,modelfile,model=sys.argv[1:]
if(todo=="train"):

	if(model=="forest"):
		trainrgbvalues=[]
		trainorientation=[]
		with open(todofile, "r") as file:
			for line in file.readlines():
				splittedline=line.split(" ")
				trainorientation.append(int(splittedline[1]))
				chunks = [int(splittedline[x]) for x in range(1, len(splittedline))]
				trainrgbvalues.append(chunks)
		trainrgbvalues=np.array(trainrgbvalues)
		
		def most_common(lst):
			return max(set(lst), key=lst.count)

		def label_count(rows):
			dict_label = {0:0,90:0,180:0,270:0}
			for row in rows:
				dict_label[row[0]] += 1
			return dict_label

		def entropy(rows):
			dict_label = label_count(rows)
			if(len(rows)==0):
				return -10
			impurity=0
			for label in dict_label:
				prob_of_label = (dict_label[label] + 1e-30)/ float(len(rows))
				impurity -= prob_of_label*np.log(prob_of_label)
			return impurity

		def info_gain(left, right, current_uncertainty):
			p = float(len(left)) / (len(left) + len(right))
			return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

		def partition(rows,colindex, valueofsplit):
			true_rows, false_rows = [],[]
			rows=np.array(rows)
			false_rows=rows[rows[:,colindex]<=valueofsplit].tolist()
			true_rows=rows[rows[:,colindex]>valueofsplit].tolist()
			return true_rows,false_rows
		def best_split(rows):
			best_col=0
			best_gain=0
			current_uncertainty=entropy(rows)
			best_var_c=128
			n_features=len(rows[0])
			for col in range(1,n_features):
				for c in [128]:
				# for c in range(1,255):    
					true_rows, false_rows = partition(rows,col,c)
					gain = info_gain(true_rows, false_rows, current_uncertainty)
					if gain >= best_gain:
						best_gain = gain
						best_col = col
						best_var_c=c
			return best_gain,best_col,best_var_c 
		def decision_tree(rows,level,randomcols):
			finalcollist={}
			gain,col,var_c=best_split(rows)
			dictofcols={}
			for i in range(len(randomcols)):
				dictofcols[i]=randomcols[i]
			true_rows, false_rows = partition(rows, col, var_c)
			orientlist=[row[0] for row in rows]
			maxorient=most_common(orientlist)
			if level==0:
				return maxorient
			finalcollist[dictofcols[col]] = {}
			finalcollist[dictofcols[col]]['var_c']=var_c
			if true_rows==[]:
				return maxorient
			else:
				finalcollist[dictofcols[col]][1] = decision_tree(true_rows, level-1,randomcols)
			if false_rows==[]:
				return maxorient
			else:
				finalcollist[dictofcols[col]][0] = decision_tree(false_rows, level-1,randomcols)
			return finalcollist
		def random_forest(trainrgbvalues,n_trees,trainorientation,level,n_cols,n_rows):
			decision=[]
			finalproductoftrees={}
			for tree in range(n_trees):
				# print(tree)
				randomrows= np.random.randint(1,len(trainrgbvalues),n_rows)
				randomcols= np.append(0,np.random.randint(1,len(trainrgbvalues[0]),n_cols))
				subrows=[ trainrgbvalues[i] for i in randomrows]
				subrows=[ [row[col] for col in randomcols] for row in subrows]
				returnedval=decision_tree(subrows,level,randomcols)
				finalproductoftrees[tree]=returnedval
			return finalproductoftrees

		# n_trees,level,n_cols,n_rows=20,6,14, 26000
		n_trees,level,n_cols,n_rows=20,6,14, 3500

		answer=random_forest(trainrgbvalues,n_trees,trainorientation,level,n_cols,n_rows)
		with open(modelfile,"w") as file2:
			file2.write(str(answer))
			file2.close()
	
	if model=="nearest" or model=="best":
		# Read training data as a dataframe
		train_data = pd.read_csv(todofile,sep=' ',header=None)
		# Generate column labels for the dataframe
		feature_names = [num for num in range(1,193)]
		labels = ['label','y']
		labels.extend(feature_names)
		# Assign column names to the train data frame
		train_data.columns = labels
		
		# Save dataframe to file
		train_data.to_csv(modelfile,index=False)
		
	if model=="adaboost":
		np.random.seed(777)
		# Read training data as a dataframe
		train_data = pd.read_csv(todofile,sep=' ',header=None)
		# Generate column labels for the dataframe
		feature_names = [num for num in range(1,193)]
		labels = ['label','y']
		labels.extend(feature_names)
		# Assign column names to the train data frame
		train_data.columns = labels

		# Add a weight column to the training samples
		train_data['w'] = [1/len(train_data) for i in range(len(train_data))]

		features = [x for x in range(1,193)]
		feature_combinations = list(itertools.combinations(features,2))
		feature_space_range = [i for i in range(len(feature_combinations))]

		for i in range(1,300):
			#pdb.set_trace()
			pair_no = np.random.choice(feature_space_range,1)[0]
			col1 = feature_combinations[pair_no][0]
			col2 = feature_combinations[pair_no][1]
			train_data[i] = train_data[col1] - train_data[col2]
			
		model_0 = AdaBoost(interest_label=0,n_estimators=50)
		model_90 = AdaBoost(interest_label=90,n_estimators=50)
		model_180 = AdaBoost(interest_label=180,n_estimators=30)
		model_270 = AdaBoost(interest_label=270,n_estimators=30)

		learner_0 = model_0.train(train_data.copy())
		learner_90 = model_90.train(train_data.copy())
		learner_180 = model_180.train(train_data.copy())
		learner_270 = model_270.train(train_data.copy())

		with open(modelfile,'wb') as f:
			pickle.dump([learner_0,learner_90,learner_180,learner_270],f)
		
	
if(todo=="test"):

	if(model=="forest"):
		testrgbvalues=[]
		testorientation=[]
		with open(todofile, "r") as file:
			for line in file.readlines():
				splittedline=line.split(" ")
				testorientation.append(int(splittedline[1]))
				chunks = [int(splittedline[x]) for x in range(1, len(splittedline))]
				testrgbvalues.append(chunks)
		with open(modelfile,"r") as file2:
			answer=file2.read()
		answer=ast.literal_eval(answer)
		def func(tree, testlist):
			for x1,x2 in enumerate(tree.values()):
				z = x2
			while (type(z)==dict):
				for y in tree.keys():
					p = y
				k=p
				valofsplitting=tree[k]['var_c']
				if(testlist[k]>valofsplitting):
					tree=tree[k][1]
				if(testlist[k]<=valofsplitting):
					tree=tree[k][0]
				if(type(tree)==dict):
					for x1,x2 in enumerate(tree.values()):
						z = x2
				z=tree                
			classs=z 
			return classs
		n_trees,level,n_cols,n_rows=20,6,14, 26000
		# n_trees,level,n_cols,n_rows=20,6,14, 3500
		testpredict=[]
		for testlist in testrgbvalues:
			dict_label = {0:0,90:0,180:0,270:0}
			for tree in answer:
				# print("tree is :",type(tree))
				tempval=func(answer[tree],testlist)
				dict_label[tempval] += 1
			max_key = max(dict_label, key=lambda k: dict_label[k])
			testpredict.append(max_key)
		count=0
		for i in range(len(testpredict)):
			if(testpredict[i]==testorientation[i]):
				count+=1
		accuracy=count/len(testpredict)
		print("Accuracy achieved is : ",accuracy)
	
	if model=="nearest" or model=="best":
		train_data = pd.read_csv(modelfile)
	
		feature_names = [num for num in range(1,193)]
		labels = ['label','y']
		labels.extend(feature_names)
			
		# Read testing data as a dataframe
		test_data = pd.read_csv(todofile,sep=' ',header=None)
		# Assign column names to the test data frame
		test_data.columns = labels
		
		# Initialize the knn model with the number of nearest neighbors and the distance metric to be used
		knn_model = KNN(11,'l2')
		#pdb.set_trace()
		# Get the predictions on the testing data
		predictions = knn_model.predict(train_data.iloc[:,1:].values.astype(np.float),test_data.iloc[:,1:].values.astype(np.float))

		# Print the score for the KNN model
		# print('Actual labels = ',list(test_data.loc[:9,'y'].values))
		print("Score for KNN = ",score(predictions,list(test_data.iloc[:,1].values)))
		
		# Write results to output file
		with open('output.txt','w') as file:
			for i in range(len(predictions)):
				file.write(test_data.iloc[i,0]+" "+str(predictions[i])+"\n")
				
	if model=="adaboost":
		np.random.seed(777)
		
		# Generate column labels for the dataframe
		feature_names = [num for num in range(1,193)]
		labels = ['label','y']
		labels.extend(feature_names)
		
		features = [x for x in range(1,193)]
		feature_combinations = list(itertools.combinations(features,2))
		feature_space_range = [i for i in range(len(feature_combinations))]

		# Read test data and pre-process it.
		test_data = pd.read_csv('test-data.txt',sep=' ',header=None)
		feature_names = [num for num in range(1,193)]
		test_data.columns = labels

		for i in range(1,300):
			#pdb.set_trace()
			pair_no = np.random.choice(feature_space_range,1)[0]
			col1 = feature_combinations[pair_no][0]
			col2 = feature_combinations[pair_no][1]
			test_data[i] = test_data[col1] - test_data[col2]
			
		model_0 = AdaBoost(interest_label=0,n_estimators=100)
		model_90 = AdaBoost(interest_label=90,n_estimators=100)
		model_180 = AdaBoost(interest_label=180,n_estimators=100)
		model_270 = AdaBoost(interest_label=270,n_estimators=100)
		
		# Load learners from file
		with open(modelfile, 'rb') as f:
			learners = pickle.load(f)
			
		learners_0 = learners[0]
		learners_90 = learners[1]
		learners_180 = learners[2]
		learners_270 = learners[3]
		
		result_m0 = model_0.predict(learners_0,test_data)
		result_m90 = model_90.predict(learners_90,test_data)
		result_m180 = model_180.predict(learners_180,test_data)
		result_m270 = model_270.predict(learners_270,test_data)

		results_list = [result_m0,result_m90,result_m180,result_m270]
		result_df = pd.concat([pd.Series(x) for x in results_list], axis=1)
		result_df.columns = [0,90,180,270]
		
		predictions = result_df.idxmax(axis=1)
		
		print("Score = ",score(predictions.values,test_data.loc[:,'y'].values))
		
		# Write results to output file
		with open('output.txt','w') as file:
			for i in range(len(predictions.values)):
				file.write(test_data.iloc[i,0]+" "+str(predictions.values[i])+"\n")
