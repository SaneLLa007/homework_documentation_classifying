import numpy as np
import math
import logging
MAXITER = 12


def pascal_triangle():
    line = [1]
    while True:
        line.append(0)
        new_line = [1]
        for i in range(len(line) - 1):
            new_line.append(line[i] + line[i + 1])
        line = new_line
        yield line


def default_moment(order: int, xl: float, xr: float, a: float = 0.0, b: float = 1.0, alpha: float = 0.0,
                   beta: float = 0.0):
    if alpha != 0:
        return ((xr - a) ** (order + 1 - alpha) - (xl - a) ** (order + 1 - alpha)) / (order + 1 - alpha)
    else:
        return ((b - xr) ** (order + 1 - beta) - (b - xl) ** (order + 1 - beta)) / (order + 1 - beta)


def Vandermonde_matrix(nodes):
    return np.array([[x ** j for j in range(len(nodes))] for x in nodes]).transpose()


def moments(max_s: int, xl: float, xr: float, a: float = 0.0, b: float = 1.0, alpha: float = 0.0, beta: float = 0.0):
    """
    compute moments of the weight 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    max_s : highest required order
    xl : left limit
    xr : right limit
    a : weight parameter a
    b : weight parameter b
    alpha : weight parameter alpha
    beta : weight parameter beta
    """

    assert alpha * beta == 0, \
        f'alpha ({alpha}) and/or beta ({beta}) must be 0'

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    m0 = default_moment(0, xl, xr, a, b, alpha, beta)
    if alpha == 0:
        a = b
        m0 = -m0

    mu = [m0]
    new_line_in_triangle = pascal_triangle()

    for i in range(1, max_s + 1):
        # коэффициенты берем из бинома ньютона
        koefs_pascal = next(new_line_in_triangle)
        powers = 1


        # рассчитаем моменты для mi
        if beta != 0:
            if i % 2 == 0:
                mi = -1 * default_moment(i, xl, xr, beta=beta, b=b)
            else:
                mi = default_moment(i, xl, xr, beta=beta, b=b)
        else:
            mi = default_moment(i, xl, xr, alpha=alpha, a=a)


        for j in range(i):
            powers *= a
            if j % 2 == 0:
                ki = powers * koefs_pascal[j + 1]
            else:
                ki = -1 * powers * koefs_pascal[j + 1]
            mi += ki * mu[-j - 1]

        mu.append(mi)

    return np.array(mu)


def quad(f, xl: float, xr: float, nodes, *params):
    """
    small Newton—Cotes formula
    f: function to integrate
    xl : left limit
    xr : right limit
    nodes: nodes within [xl, xr]
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mu = moments(len(nodes) - 1, xl, xr, *params)

    vandermonde_matrix = Vandermonde_matrix(nodes)

    vector_Ai = np.linalg.solve(vandermonde_matrix, mu)

    answer = 0
    for i in range(len(nodes)):
        answer += vector_Ai[i] * f(nodes[i])
    return answer


def quad_gauss(f, xl: float, xr: float, n: int, *params):
    """
    small Gauss formula
    f: function to integrate
    xl : left limit
    xr : right limit
    n : number of nodes
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mu = np.array(moments(2 * n - 1, xl, xr, *params))

    vector_b = - mu[n:]
    matrix_A = np.zeros((n, n))

    for i in range(n):
        matrix_A[i] = mu[i:n + i]

    vector_ai = np.linalg.solve(matrix_A, vector_b)  # вектор коэффициентов

    vector_ai = np.append(vector_ai, 1)

    vector_ai_reversed = np.flip(vector_ai)

    nodes = np.roots(vector_ai_reversed)

    vandermonde_matrix = Vandermonde_matrix(nodes)

    vector_Ai = np.linalg.solve(vandermonde_matrix, mu[:n])

    answer = 0
    for i in range(len(nodes)):
        answer += vector_Ai[i] * f(nodes[i])
    return answer


def composite_quad(f, xl: float, xr: float, N: int, n: int, *params):
    """
    composite Newton—Cotes formula
    f: function to integrate
    xl : left limit
    xr : right limit
    N : number of steps
    n : number of nodes od small formulae
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mesh = np.linspace(xl, xr, N + 1)
    return sum(quad(f, mesh[i], mesh[i + 1], equidist(n, mesh[i], mesh[i + 1]), *params) for i in range(N))


def composite_gauss(f, a: float, b: float, N: int, n: int, *params):
    """
    composite Gauss formula
    f: function to integrate
    xl : left limit
    xr : right limit
    N : number of steps
    n : number of nodes od small formulae
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mesh = np.linspace(a, b, N + 1)
    return sum(quad_gauss(f, mesh[i], mesh[i + 1], n, *params) for i in range(N))


def equidist(n: int, xl: float, xr: float):
    if n == 1:
        return [0.5 * (xl + xr)]
    else:
        return np.linspace(xl, xr, n)


def runge(s1: float, s2: float, L: float, m: float):
    """ estimate m-degree error for s2 """
    return (s2 - s1) / (L**m - 1)


def aitken(s1: float, s2: float, s3: float, L: float):
    """
    estimate convergence order
    s1, s2, s3: consecutive composite quads
    return: convergence order estimation
    """
    estimate = (s2 - s1) / (s3 - s2)

    if estimate < 0:
        return -1
    else:
        return math.log(estimate, L)


def doubling_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with theoretical convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 1
    N = 1

    if n % 2 != 0:
        m = n
    else:
        m = n - 1
    m += 1


    s1 = composite_quad(f, xl, xr, N, n, *params)

    # цикл по итерациям
    while iter != MAXITER+1:
        N *= 2

        s2 = composite_quad(f, xl, xr, N, n, *params)

        # берём погрешность из метода рунге
        estimate = np.abs(runge(s1, s2, 2, m))

        if estimate >= tol:
            s1 = s2
        else:
            return N, s2, estimate

        iter += 1

    if iter == MAXITER:
        print("Convergence not reached!")
        return -1


def doubling_nc_aitken(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with Aitken estimation of the convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # m : estimated convergence rate by Aitken for S
    # iter : number of iterations (steps doubling)
    iter = 1
    N = 100

    # считаем интеграл с шагом h
    s1 = composite_quad(f, xl, xr, N, n, *params)
    N *= 2
    s2 = composite_quad(f, xl, xr, N, n, *params)
    # цикл по итерациям
    while iter != MAXITER + 1:
        N *= 2
        # считаем интеграл с шагом h/2
        s3 = composite_quad(f, xl, xr, N, n, *params)

        m = aitken(s1, s2, s3, 2)

        # берём погрешность из метода рунге
        estimate = np.abs(runge(s2, s3, 2, m))
        if estimate >= tol:
            s1 = s2
            s2 = s3
        else:
            return N, s3, estimate, m

        iter += 1

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10 * tol, -100


def doubling_gauss(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with theoretical convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 1
    N = 10
    m = (2 * n - 1) + 1

    # считаем интеграл с шагом h
    s1 = composite_gauss(f, xl, xr, N, n, *params)
    # цикл по итерациям
    while iter != MAXITER + 1:
        N *= 2
        # считаем интеграл с шагом h/2
        s2 = composite_gauss(f, xl, xr, N, n, *params)

        # берём погрешность из метода рунге
        estimate = np.abs(runge(s1, s2, 2, m))
        if estimate >= tol:
            s1 = s2
        else:
            return N, s2, estimate

        iter += 1

    if iter == MAXITER:
        print("Convergence not reached!")
        return -1


def doubling_gauss_aitken(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with Aitken estimation of the convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # m : estimated convergence rate by Aitken for S
    # iter : number of iterations (steps doubling)
    iter = 1
    N = 10

    s1 = composite_gauss(f, xl, xr, N, n, *params)
    N *= 2
    s2 = composite_gauss(f, xl, xr, N, n, *params)

    # цикл по итерациям
    while iter != MAXITER + 1:
        N *= 2
        # считаем интеграл с шагом h/2
        s3 = composite_gauss(f, xl, xr, N, n, *params)

        m = aitken(s1, s2, s3, 2)

        # берём погрешность из метода рунге
        estimate = np.abs(runge(s2, s3, 2, m))
        if estimate >= tol:
            s1 = s2
            s2 = s3
        else:
            return N, s3, estimate, m

        iter += 1

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10 * tol, -100


def optimal_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """ estimate the optimal step with Aitken and Runge procedures
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 1
    N = 200
    h_old = (xr - xl) / (4*N)

    s1 = composite_quad(f, xl, xr, N, n, *params)
    N *= 2
    s2 = composite_quad(f, xl, xr, N, n, *params)
    N *= 2
    s3 = composite_quad(f, xl, xr, N, n, *params)

    L = 2
    m = aitken(s1, s2, s3, L)


    # если m далеко от теоретического значения, значит мы находимся в области,
    # где отброшенные величины не малы и нам нужно продолжать делить шаг
    while m > 4.5 or m < 2.5:
        s1 = s2
        s2 = s3
        N *= 2
        s3 = composite_quad(f, xl, xr, N, n, *params)
        m = aitken(s1, s2, s3, L)

    # далее предлагаем идти по оптимальному шагу
    coef = 0.95
    h_new = coef * h_old * (tol / np.abs(runge(s2, s3, L, m))) ** (1 / m)

    N = int((xr - xl) / h_new) + 1

    while iter != MAXITER + 1:
        s1 = s2
        s2 = s3

        s3 = composite_quad(f, xl, xr, N, n, *params)

        m = aitken(s1, s2, s3, L)

        # берём погрешность из метода рунге
        estimate = np.abs(runge(s2, s3, L, m))

        if estimate < tol:
            return N, s3, estimate

        N *= 2
        iter += 1

    print("Convergence not reached!")
    return 0, 0, 10 * tol, -100

