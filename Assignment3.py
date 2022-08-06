# CSCI 323/700
# Summer 2022
# Assignment 3 - Empirical Performance of Matrix Multiplication
# Andrew Zheng


import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

assign_num = 3


def random_matrix(mn, mx, rows, cols):
    return [[random.randint(mn, mx) for col in range(0, cols)] for row in (range(0, rows))]


def one_matrix(mn, mx, rows, cols):
    return [[1 for col in range(0, cols)] for row in (range(0, rows))]


def print_matrix(matrix):
    print('\n'.join([''.join(['{:4}'.format(element) for element in row])
                     for row in matrix]) + "\n")


def native_mult(m1, m2):
    return np.dot(m1, m2)


def simple_mult(m1, m2):
    rows = len(m1)
    cols = len(m2[0])
    m3 = [[0 for x in range(cols)]
          for y in range(rows)]

    for i in range(rows):
        for j in range(cols):
            m3[i][j] = 0
            for x in range(cols):
                m3[i][j] += (m1[i][x] *
                             m2[x][j])
    return m3


# Revised based on https://www.interviewbit.com/blog/strassens-matrix-multiplication/

def strassen_mult(m1, m2):
    n = m1.shape[0]

    if n == 1:
        return m1 * m2

    nx = int(2 ** np.ceil(np.log2(n)))

    result = np.zeros((nx, nx))

    temp = np.zeros((nx, nx))
    temp[:n, :n] = m1
    m1 = temp
    temp = np.zeros((nx, nx))
    temp[:n, :n] = m2
    m2 = temp

    m = nx // 2

    a11 = m1[:m, :m]
    a12 = m1[:m, m:]
    a21 = m1[m:, :m]
    a22 = m1[m:, m:]

    b11 = m2[:m, :m]
    b12 = m2[:m, m:]
    b21 = m2[m:, :m]
    b22 = m2[m:, m:]

    p1 = strassen_mult(a11 + a22, b11 + b22)
    p2 = strassen_mult(a21 + a22, b11)
    p3 = strassen_mult(a11, b12 - b22)
    p4 = strassen_mult(a22, b21 - b11)
    p5 = strassen_mult(a11 + a12, b22)
    p6 = strassen_mult(a21 - a11, b11 + b12)
    p7 = strassen_mult(a12 - a22, b21 + b22)

    result[:m, :m] = p1 + p4 - p5 + p7
    result[:m, m:] = p3 + p5
    result[m:, :m] = p2 + p4
    result[m:, m:] = p1 - p2 + p3 + p6
    result = result[:n, :n]

    return result


def plot_time(dict_algs, sizes, algs, trials):
    alg_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for alg in algs:
        alg_num += 1
        d = dict_algs[alg.__name__]
        x_axis = [j + 0.05 * alg_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.05, alpha=0.75, label=alg.__name__)
    plt.legend()
    plt.title("Run time of search algorithms")
    plt.xlabel("Different sizes of n")
    plt.ylabel("Time for " + str(trials) + "one trial (ms)")
    plt.savefig("Assignment " + str(assign_num) + ".png")
    plt.show()


def main():
    sizes = [10 * i for i in range(1, 11)]
    trials = 1
    algs = [native_mult, simple_mult, strassen_mult]
    # m3=np.array

    dict_algs = {}
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):

            m1 = np.array(random_matrix(-1, 1, size, size))
            m2 = random_matrix(-1, 1, size, size)

            # m1 = np.array(one_matrix(-1, 1, size, size))
            # m2 = one_matrix(-1, 1, size, size)
            # print_matrix(m1)
            # print_matrix(m2)
            for alg in algs:
                start_time = time.time()
                #  a = np.array(m1)
                #  b = np.array(m2)
                m3 = alg(m1, m2)

                # mult = strassen_mult(m1, m2)

                end_time = time.time()

                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 95 * net_time

        # print_matrix(m3)
    # print(dict_searches)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(dict_algs).T
    print(df)
    plot_time(dict_algs, sizes, algs, trials)


if __name__ == "__main__":
    main()
