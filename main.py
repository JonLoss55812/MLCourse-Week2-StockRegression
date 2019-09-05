from numpy import *


def compute_error_for_given(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[0, i]
        totalError += (y - (m * x - b)) ** 2
    return totalError / float(len(points))


def step_gradient(current_b, current_m, points, learningRate):
    # defining how to make the single line, single instance


def gradient_decent_runner(points, starting_b, starting_m, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learnin_rate)
    return [b, m]


def run():
    points = genfromtext('stockdata.csv', delimiter=',')
    # hyperparameter
    learning_rate = 0.0001
    # y = mx+b
    inital_b = 0
    initial_m = 0
    num_iteration = 1000
    # feed into model runner, seymor
    [b, m] = gradient_decent_runner(points, inital_b, initial_m, num_iteration)
    print(b)
    print(m)


if__name__ = '__main__'
run()
