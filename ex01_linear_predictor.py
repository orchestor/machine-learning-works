import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.  Load data
data = pd.read_csv('ex1data1.txt', names = ['population', 'profit'])
print data.head()
X_df = pd.DataFrame(data.population)
y_df = pd.DataFrame(data.profit)


# 2. Plot data
print(plt.figure(figsize=(10,8)))
print(plt.plot(X_df, y_df, 'kx'))
print(plt.xlabel('Population of City in 10,000s'))
print(plt.ylabel('Profit in $10,000s'))
iterations = 1500
learning_rate = 0.01
m = len(y)
X_df['intercept'] = 1


# 3. compute the cost function
def cost_function(X, y, theta):
    m = len(y)
    J = np.sum((X.dot(theta)-y)**2)/2/m
    return J
print("The cost is \n", cost_function(X, y, theta))


# 4. Caculate weights and each step cost
def compute_theta_cost(X, y, theta, learning_rate, iterations):
    cost_history = [0]*iterations

    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - y
        gradient = X.T.dot(loss)/m
        theta = theta - learning_rate*gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost
    return theta, cost_history
(t, c) = compute_theta_cost(X,y,theta,learning_rate, iterations)
print(t)


# 5. Make prediction based on weights
print np.array([3.5, 1]).dot(t)
print np.array([7, 1]).dot(t)


# 6. Plot the best fit line
best_fit_x = np.linspace(0, 25, 20)
best_fit_y = [t[1] + t[0]*xx for xx in best_fit_x]
plt.figure(figsize=(10,6))
plt.plot(X_df.population, y_df, '.')
print(plt.plot(best_fit_x, best_fit_y, '-'))
plt.axis([0,25,-5,25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population with Linear Regression Line')
