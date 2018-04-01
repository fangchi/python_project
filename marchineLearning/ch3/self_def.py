# -*- coding: utf-8 -*

# object likelihood function
import numpy as np


def likelihood_sub(x, y, beta):
    '''
    @param X: one sample variables
    @param y: one sample label
    @param beta: the parameter vector in 3.27
    @return: the sub_log-likelihood of 3.27
    '''
    return -y * np.dot(beta, x.T) + np.math.log(1 + np.math.exp(np.dot(beta, x.T)))


def likelihood(X, y, beta):
    '''
    @param X: the sample variables matrix
    @param y: the sample label matrix
    @param beta: the parameter vector in 3.27
    @return: the log-likelihood of 3.27
    '''
    sum = 0
    m, n = np.shape(X)

    for i in range(m):
        sum += likelihood_sub(X[i], y[i], beta)

    return sum


def partial_derivative(X, y, beta):  # refer to 3.30 on book page 60
    '''
    @param X: the sample variables matrix
    @param y: the sample label matrix
    @param beta: the parameter vector in 3.27
    @return: the partial derivative of beta [j]
    '''

    m, n = np.shape(X)
    pd = np.zeros(n)

    for i in range(m):
        tmp = y[i] - sigmoid(X[i], beta)
        for j in range(n):
            pd[j] += X[i][j] * (tmp)
    return pd


def gradDscent_1(X, y):  # implementation of fundational gradDscent algorithms
    '''
    @param X: X is the variable matrix
    @param y: y is the label array
    @return: the best parameter estimate of 3.27
    '''
    import matplotlib.pyplot as plt

    h = 0.1  # step length of iterator
    max_times = 500  # give the iterative times limit
    m, n = np.shape(X)

    b = np.zeros((n, max_times))  # for show convergence curve of parameter
    beta = np.zeros(n)  # parameter and initial
    delta_beta = np.ones(n) * h
    llh = 0
    llh_temp = 0

    for i in range(max_times):
        beta_temp = beta

        for j in range(n):
            # for partial derivative
            beta[j] += delta_beta[j]
            llh_tmp = likelihood(X, y, beta)
            delta_beta[j] = -h * (llh_tmp - llh) / delta_beta[j]

            b[j, i] = beta[j]

            beta[j] = beta_temp[j]

        beta += delta_beta
        llh = likelihood(X, y, beta)

    t = np.arange(max_times)

    f2 = plt.figure(3)

    p1 = plt.subplot(311)
    p1.plot(t, b[0])
    plt.ylabel('w1')

    p2 = plt.subplot(312)
    p2.plot(t, b[1])
    plt.ylabel('w2')

    p3 = plt.subplot(313)
    p3.plot(t, b[2])
    plt.ylabel('b')

    plt.show()
    return beta


def gradDscent_2(X, y):  # implementation of stochastic gradDscent algorithms
    '''
    @param X: X is the variable matrix
    @param y: y is the label array
    @return: the best parameter estimate of 3.27
    '''
    import matplotlib.pyplot as plt

    m, n = np.shape(X)  # m =  ? m= 行数  n = 列
    h = 0.5  # step length of iterator and initial 步长
    beta = np.zeros(n)  # parameter and initial
    delta_beta = np.ones(n) * h
    llh = 0
    llh_temp = 0
    b = np.zeros((n, m))  # for show convergence curve of parameter n行m列  记录结果

    for i in range(m):  #遍历所有训练集数据
        beta_temp = beta   #初始化

        for j in range(n):  #针对每一个训练集的每个参数
            # for partial derivative 偏导
            h = 0.5 * 1 / (1 + i + j)  # change step length of iterator
            beta[j] += delta_beta[j]

            b[j, i] = beta[j]

            llh_tmp = likelihood_sub(X[i], y[i], beta)
            delta_beta[j] = -h * (llh_tmp - llh) / delta_beta[j]

            beta[j] = beta_temp[j]

        beta += delta_beta
        llh = likelihood_sub(X[i], y[i], beta)

    # t = np.arange(m)

    # f2 = plt.figure(3)
    #
    # p1 = plt.subplot(311)
    # p1.plot(t, b[0])
    # plt.ylabel('w1')
    #
    # p2 = plt.subplot(312)
    # p2.plot(t, b[1])
    # plt.ylabel('w2')
    #
    # p3 = plt.subplot(313)
    # p3.plot(t, b[2])
    # plt.ylabel('b')

    # plt.show()

    print('beta:')
    print(beta);
    return beta


def sigmoid(x, beta):
    '''
    @param x: is the predict variable
    @param beta: is the parameter
    @return: the sigmoid function value
    '''
    return 1.0 / (1 + np.math.exp(- np.dot(beta, x.T)))


def predict(X, beta):
    '''
    prediction the class lable using sigmoid
    @param X: data sample form like [x, 1]
    @param beta: the parameter of sigmoid form like [w, b]
    @return: the class lable array
    '''
    m, n = np.shape(X)
    y = np.zeros(m)

    for i in range(m):
        if sigmoid(X[i], beta) > 0.5: y[i] = 1;
    return y

    return beta


if __name__ == '__main__':
    from sklearn import model_selection

    # load the CSV file as a numpy matrix
    dataset = np.loadtxt('data.csv', delimiter=",")

    # separate the data from the target attributes
    X = dataset[:, 1:3]  # 获取x变量
    y = dataset[:, 3]  # 获取结果值

    m, n = np.shape(X)

    # X_train, X_test, y_train, y_test
    np.ones(n)
    m, n = np.shape(X)  # m 行  n列 m=17 n=2
    X_ex = np.c_[X, np.ones(m)]  # extend the variable matrix to [x, 1] test_size = 0.4 表示训练集>测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_ex, y, test_size=0.4, random_state=0)

    # using gradDescent to get the optimal parameter beta = [w, b] in page-59
    beta = gradDscent_2(X_train, y_train)

    # prediction, beta mapping to the model
    y_pred = predict(X_test, beta)

    m_test = np.shape(X_test)[0]
    # calculation of confusion_matrix and prediction accuracy
    cfmat = np.zeros((2, 2))
    for i in range(m_test):
        if y_pred[i] == y_test[i] == 0:
            cfmat[0, 0] += 1
        elif y_pred[i] == y_test[i] == 1:
            cfmat[1, 1] += 1
        elif y_pred[i] == 0:
            cfmat[1, 0] += 1
        elif y_pred[i] == 1:
            cfmat[0, 1] += 1
    print("cfmat")
    print(cfmat)
