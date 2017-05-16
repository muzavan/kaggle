# main.py, May 17 2017
# using decision tree because this is a simple classification with only two class, with data both nominal and numeric 

from sklearn import tree
import csv

# training file
tr_file = "../data/train_p.csv"

train_list = list()

# copy data from training file
with open(tr_file,"rt") as file:
	train_reader = csv.reader(file, delimiter=',')
	for row in train_reader:
		train_list.append(row)
		
# create train data
train_feature_list = list()
train_class_list = list()

for itr in range(1, len(train_list)):
	train_feature_list.append(train_list[itr][1:6]) # This is dumb because we put the class in the beginning
	train_class_list.append(train_list[itr][0])
	
# Get tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_feature_list, train_class_list)

# test file
ts_file = "../data/test_p.csv"

test_feature_list = list()
test_class_list = list()

# copy data from test file
with open(ts_file,"rt") as file:
	test_reader = csv.reader(file, delimiter=',')
	for row in test_reader:
		test_feature_list.append(row)
		
# testint test_feature_list
result = clf.predict(test_feature_list[1:])

# Expected result file
exp_file = "../data/e_result.csv"

# Copy result file
with open(exp_file,"rt") as file:
	exp_reader = csv.reader(file, delimiter=',')
	for row in exp_reader:
		test_class_list.append(row)

# Take only the expected result from file
exp_result = list()
for itr in range(1,len(test_class_list)):
	exp_result.append(test_class_list[itr][-1])

# print the score
score = 0

for itr in range(0, len(exp_result)):
	if(result[itr] == exp_result[itr]):
		score = score + 1
	else:
		print("Diff : {} {}".format(result[itr],exp_result[itr]))
		

print("Result : {} of {} ({}%) are true".format(score,len(exp_result), score*100/len(exp_result)))
