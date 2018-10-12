import math
from numpy import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Adjusts gradient for worse estimations
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-1.0 * z))

# Calculates the combination of all the elements in the row along with the coefficients
def predict(row, coefficients):
    z = 0
    for i in range(len(coefficients)):
        z += coefficients[i] * row[i]
    return sigmoid(z)

# Calculates each new value and slowly optimizes function through repeated iterations
def gradient(X, Y, coefficients, learningRate):
    constant = float(learningRate) / float(len(Y))
    newCoefficient = []
    for i in range (len(coefficients)):
        sum = 0
        for j in range(len(Y)):
            currentX = X[j][i]
            currentPredict = predict(coefficients, X[j])
            error = (currentPredict - Y[j]) * currentX
            sum += error
        costError = constant * sum
        newValue = coefficients[i] - costError
        newCoefficient.append(newValue)
    return newCoefficient

# Repeats the gradient according to amount specified
def regression(X, Y, coefficients, epochs, learningRate):
    for i in range(epochs):
        newCoefficient = gradient(X, Y, coefficients, learningRate)
        coefficients = newCoefficient
    return coefficients

# Returns accuracy of algorithm
def getAccuracy(coefficients, TestX, TestY):
    score = 0
    for i in range (len(TestX)):
        prediction = round(predict(coefficients, TestX[i]))
        answer = TestY[i]
        if prediction == answer:
            score += 1
    totalScore = (float(score) / float(len(TestX))) * 100
    return totalScore

# Returns the predictions of the y values
def getPredictions(TestX, coefficients):
    predictions = []
    for i in range (len(TestX)):
        prediction = round(predict(coefficients, TestX[i]))
        predictions.append(prediction)
    return predictions

def main():
    # Load in data
    data = genfromtxt('breast-cancer-wisconsin.training.data.txt', dtype=int, delimiter=',', usecols = range(1, 11))
    test = genfromtxt('breast-cancer-wisconsin.test.data.txt', dtype=int, delimiter=',', usecols = range(1, 11))
    dataX = data[:, 0:9]
    testX = test[:, 0:9]
    dataY = data[:, 9]
    testY = test[:, 9]
    # Sets all the values from 2 or 4 to 0 or 1
    for i in range (len(data[:, 0])):
        if (data[:, 9][i] == 4):
            data[:, 9][i] = 1
        else: 
            data[:, 9][i] = 0
    for i in range (len(test[:, 0])):
        if (test[:, 9][i] == 4):
            test[:, 9][i] = 1
        else: 
            test[:, 9][i] = 0
    # Iterates through 100 times, with a learning rate of 0.1
    epochs = 100
    learningRate = 0.1
    coefficients = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Prints out data to user
    newCoefficients = regression(dataX, dataY, coefficients, epochs, learningRate)
    accuracy = getAccuracy(newCoefficients, testX, testY)
    predictions = getPredictions(testX, newCoefficients)
    print "Accuracy: ", accuracy, "%"
    print "New Coefficients: ", newCoefficients
    print "Confusion Matrix: \n", confusion_matrix(testY, predictions)
    print "Report: \n", classification_report(testY, predictions)
main()