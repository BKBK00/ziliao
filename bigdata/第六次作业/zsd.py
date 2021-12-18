import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from sklearn.model_selection import KFold


def plot_scatter(data, X_labels):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positive['X1'], positive['X2'], c='red', marker='o', label='Admitted')
    ax.scatter(negative['X1'], negative['X2'], c='green', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel(X_labels[0] + ' Score')
    ax.set_ylabel(X_labels[1] + ' Score')


def feature_mapping(X1, X2, degree):
    data = pd.DataFrame()
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            data['F' + str(i) + str(j)] = np.power(X1, i - j) * np.power(X2, j)
    return data


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def costReg(theta, x, y, learningRate):
    theta = np.matrix(theta)

    X = np.matrix(x)
    y = np.matrix(y).T
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, x, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(x)
    y = np.matrix(y).T

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for j in range(parameters):
        term = np.multiply(error, X[:, j])

        if j == 0:
            grad[j] = np.sum(term) / len(X)
        else:
            grad[j] = (np.sum(term)) / len(X) + ((learningRate / len(x)) * theta[:, j])
    return grad


def find_decision_boundary(density, degree, theta, threshold, cord_bounds):
    t1 = np.linspace(cord_bounds[0], cord_bounds[1], density)
    t2 = np.linspace(cord_bounds[2], cord_bounds[3], density)

    coordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*coordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, degree)
    mapped_cord.insert(0, 'Ones', 1)
    print(mapped_cord.shape)
    print(theta.shape)

    inner_product = np.matrix(mapped_cord) * theta.T
    decision = mapped_cord[np.abs(inner_product) < threshold]
    print(decision)

    return decision.F10, decision.F11


def draw_boundary(data, degree, theta, x_label, y_label, cord_bounds):
    density = 1000
    threshold = 2 * 10 ** -3

    x, y = find_decision_boundary(density, degree, theta, threshold, cord_bounds)

    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    plot_scatter(data, x_label)
    plt.scatter(x, y, c='blue', s=10)
    plt.title('Decision boundary')


# read data
df1 = pd.read_csv('train.txt', sep='\t', engine='python')
df2 = pd.read_csv('test.txt', sep='\t', engine='python')
x1_train, x2_train, y_train = df1['X1'], df1['X2'], df1['y']
x1_test, x2_test, y_test = df2['X1'], df2['X2'], df2['y']

# get mapped_x and y
degree = 6
train_data_map = feature_mapping(x1_train, x2_train, degree)
test_data_map = feature_mapping(x1_test, x2_test, degree)

train_data_map.insert(0, 'Ones', 1)
test_data_map.insert(0, 'Ones', 1)

x_train = np.array(train_data_map.values)
x_test = np.array(test_data_map.values)

y_train = np.array(y_train.values)
y_test = np.array(y_test.values)
theta = np.zeros(((degree + 1) * (degree + 2)) // 2)
learningRate = 1

# print(costReg(theta, x_train, y_train, learningRate))
# draw boundary
train_results = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x_train, y_train, 1))
draw_boundary(df1, degree, np.matrix(train_results[0]), ['x_1', 'x_2'], 'accepted', [-1, 1.5, -1, 1.5])
plt.show()

# get the plot of theta when lambda changes
lambdas = list(np.array(pd.read_csv('lambda.txt', header=None).values).flatten())
train_results = []
test_results = []
for i in lambdas:
    train_results.append(opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x_train, y_train, i)))
    test_results.append(opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x_test, y_test, i)))

log_lambdas = [np.log10(i) for i in lambdas]
train_thetas = [train_results[i][0] for i in range(len(train_results))]
test_thetas = [test_results[i][0] for i in range(len(test_results))]

plt.plot(log_lambdas, train_thetas, linestyle='--')
plt.xlabel('log10(lambda)')
plt.ylabel('theta')
plt.show()

# get accuracy set
train_accuracy = []
test_accuracy = []
for i in range(len(train_results)):
    train_theta_min = np.matrix(train_results[i][0])
    test_theta_min = np.matrix(test_results[i][0])
    train_predictions = predict(train_theta_min, x_train)
    test_predictions = predict(test_theta_min, x_test)
    train_correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in
                     zip(train_predictions, y_train)]
    test_correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in
                    zip(test_predictions, y_test)]
    train_accuracy.append(int(str(train_correct).count('1')) / len(train_correct))
    test_accuracy.append(int(str(test_correct).count('1')) / len(test_correct))

# get the plot of accuracy on train set when lambda changes
plt.plot(log_lambdas, train_accuracy, c='blue')
plt.xlabel('log10(lambda)')
plt.ylabel('accuracy on train data set')
plt.show()

# get the plot of accuracy on test set when lambda changes
plt.plot(log_lambdas, test_accuracy, c='green')
plt.xlabel('log10(lambda)')
plt.ylabel('accuracy on test data set')
plt.show()

# get the plot of accuracy on 10 fold Validation
kf = KFold(n_splits=10)
fold_accuracies = []
for train, test in kf.split(x_train):
    fold_results = []
    for i in lambdas:
        fold_results.append(
            opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x_train[train], y_train[train], i)))
    fold_accuracy = []
    for j in range(len(fold_results)):
        fold_theta_min = np.matrix(fold_results[j][0])
        fold_predictions = predict(fold_theta_min, x_train[test])
        train_correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in
                         zip(fold_predictions, y_train[test])]
        fold_accuracy.append(int(str(train_correct).count('1')) / len(train_correct))
    fold_accuracies.append(fold_accuracy)
    fold_results.clear()

fold_accuracies_mean = np.mean(fold_accuracies, 0)
plt.plot(log_lambdas, fold_accuracies_mean, c='red')
plt.xlabel('log10(lambda)')
plt.ylabel('accuracy on 10-fold cv data set')
plt.show()
