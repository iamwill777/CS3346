import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Used ID3 Heuristic
def majority(attributes, data, target):
    frequency = {}
    index = attributes.index(target)
    for tuple in data:
        if (frequency.has_key(tuple[index])):
            frequency[tuple[index]] += 1 
        else:
            frequency[tuple[index]] = 1
    max = 0
    majority = ""
    for key in frequency.keys():
        if frequency[key]>max:
            max = frequency[key]
            majority = key
    return majority
def entropy(attributes, data, target):
    frequency = {}
    currentEntropy = 0.0
    #find index of the target attributes
    i = attributes.index(target)
    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (frequency.has_key(entry[i])):
            frequency[entry[i]] += 1.0
        else:
            frequency[entry[i]]  = 1.0
    # Calculate the entropy of the data for the target attribute
    for freq in frequency.values():
        currentEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
    return currentEntropy
def gain(attributes, data, attribute, target):
    frequency = {}
    currentEntropy = 0.0
    # Index of attribute
    i = attributes.index(attribute)
    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (frequency.has_key(entry[i])):
            frequency[entry[i]] += 1.0
        else:
            frequency[entry[i]]  = 1.0
    # Calculates sum of entropies according to probability occurring
    for value in frequency.keys():
        valProb = frequency[value] / sum(frequency.values())
        dataSubset = []
        for j in range(len(data)):
            if data[j][i] == value:
                dataSubset.append(entry)
        currentEntropy += valProb * entropy(attributes, dataSubset, target)
    return entropy(attributes, data, target) - currentEntropy
# Chooses best attribute according to information gain
def chooseAttribute(data, attributes, target):
    best = None
    max = 0
    for attribute in attributes:
        newGain = gain(attributes, data, attribute, target)
        if newGain > max:
            max = newGain
            best = attribute
    return best
#Get values in the column of the given attribute 
def getValues(data, attributes, attribute):
    index = attributes.index(attribute)
    values = []
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values
# Gets example
def getExamples(data, attributes, best, value):
    examples = [[]]
    index = attributes.index(best)
    for entry in data:
        if (entry[index] == value):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            examples.append(newEntry)
    examples.remove([])
    return examples
# Builds a decision tree recursively
def buildTree(data, attributes, target, recursion):
    recursion += 1
    # Gets the data and values
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majority(attributes, data, target)
    # Base case
    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Finds best attribute to classify data
        best = chooseAttribute(data, attributes, target)
        tree = {best:{}}
        # Creates a subtree based on best attribute
        for value in getValues(data, attributes, best):
            examples = getExamples(data, attributes, best, value)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = buildTree(examples, newAttr, target, recursion)
            # Add new subtree in
            tree[best][value] = subtree
    return tree
    
def main():
    # Opens both training and test files and finds the attributes and target
    data = []
    test = []
    attributes = ["origin", "age", "gender", "education", "degree", "language", "job_level", "current_role", "hired"]
    target = attributes[8]
    trainingFile = "Assignment 4 - Question 3 training data.csv"
    testFile = "Assignment4 - Question 3  test_data.csv"
    with open(trainingFile) as file:
        #skip the first line
        file.readline()
        for line in file.readlines():
            origin, age, gender, education, degree, language, job_level, current_role, hired = line.strip().split(',')
            age = int(age)
            data.append([origin, age, gender, education, degree, language, job_level, current_role, hired])
    with open(testFile) as file:
        #skip the first line
        file.readline()
        for line in file.readlines():
            origin, age, gender, education, degree, language, job_level, current_role, hired = line.strip().split(',')
            age = int(age)
            test.append([origin, age, gender, education, degree, language, job_level, current_role, hired])
    # Builds tree
    decisionTree = buildTree(data, attributes, target, 0)
    print(decisionTree)
main()