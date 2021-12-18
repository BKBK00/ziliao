import numpy as np


def myFitness(p):

    x1, y1, x2, y2 = p
    X = np.array([9,9,8,3,0,4,4,2,0,5])
    Y = np.array([9,3,6,9,6,2,0,0,9,2])
    sum=0
    a=0
    i=0
    while i<=9:
        facility1_dist = abs(x1 -X[i]) + abs(y1 - Y[i])
        facility2_dist = abs(x2 - X[i]) + abs(y2 -Y[i])
        if facility1_dist > facility2_dist:
            a = facility2_dist
        else:
            a = facility1_dist
        sum+=a
        i+=1
    return sum

from sko.GA import GA

ga = GA(func=myFitness, n_dim=4, size_pop=10, max_iter=60, prob_mut=0.01, lb=[0,0,0,0], ub=[10,10,10,10], precision=[1,1,1,1])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

X = np.array([9, 9, 8, 3, 0, 4, 4, 2, 0, 5])
Y = np.array([9, 3, 6, 9, 6, 2, 0, 0, 9, 2])

import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
plt.figure(figsize=(7, 9))

plt.subplot(311)
plt.plot(Y_history.index, Y_history.values, '.', color='red')
plt.title('fig_1 history values\n fig_2  Convergence trend \nfig_3 best output')

plt.subplot(312)
Y_history.min(axis=1).cummin().plot(kind='line')

plt.subplot(313)

i=0
while i<=9:
        facility1_dist = abs(best_x[0] -X[i]) + abs(best_x[1] - Y[i])
        facility2_dist = abs(best_x[2] - X[i]) + abs(best_x[3] -Y[i])
        if facility1_dist > facility2_dist:
            plt.plot([X[i], best_x[2]], [Y[i], best_x[3]], color='r')
            print("营业点（{},{}）的配送站为2号配送站（{},{}）".format(X[i],Y[i],best_x[0],best_x[1]))
        else:
            plt.plot([X[i], best_x[0]], [Y[i], best_x[1]], color='b')
            print("营业点（{},{}）的配送站为1号配送站（{},{}）".format(X[i],Y[i],best_x[2],best_x[3]))
        i+=1


plt.scatter(X,Y,color='b')
plt.scatter(best_x[0],best_x[1],color='r')
plt.scatter(best_x[2],best_x[3],color='g')

plt.show()