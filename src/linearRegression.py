from numpy import *
import matplotlib.pyplot as plt

# The gradient function of this problem
def linearGradient(x, y, b, m, epochs, learningRate):
    # Calculates the slope and y-intercept of graph
    n = float(len(y))
    # Loops according to amount of times given
    for i in range (epochs):
        # Resets gradient for each following iteration
        bGradient = 0
        mGradient = 0
        for j in range(0, len(y)):
            currentx = x[j]
            currenty = y[j]
            # Calculates gradient and adds it one by one to the slope
            bGradient += -(2/n) * (currenty - ((m * currentx) + b))
            mGradient += -(2/n) * currentx * (currenty - ((m * currentx) + b))
        m = m - (learningRate * mGradient)
        b = b - (learningRate * bGradient)
    return [m, b]

def plotRegressionLine(x, y, m, b):
    # Plot as scatter graph
    plt.scatter(x, y, color = "m", marker = "o", s = 30)
    # The regression line
    line = m*x + b
    # Plotting the regression line
    plt.plot(x, line, color = "g")
    # Putting axis labels
    plt.xlabel('X')
    plt.ylabel('Y')
    # Show plot
    plt.show()
 
def main():
    # Load in data
    data = genfromtxt('data.csv', dtype=float, delimiter=',', skip_header = 1) 
    # Get x and y, initialize m and b
    x = data[:, 0]
    y = data[:, 1]
    b = 0
    m = 0
    # Run it 1000 times with a learning rate of 0.001
    epochs = 1000
    learningRate = 0.0001
    newM, newB = linearGradient(x, y, b, m, epochs, learningRate)
    # Print answer and also graph
    print"New B, New M:", newB,",", newM
    plotRegressionLine(x, y, newM, newB)

main()