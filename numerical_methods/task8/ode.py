import numpy as np
from dataclasses import dataclass

a = np.array([[0,0,0],[1,2,3],[4,6,5]])
print(a)
index = np.argmax(a[1:,1:])
print(index)
indx = np.unravel_index(np.argmax(a[1:,1:]),a[1:,1:].shape)
print(np.argmax(a))
print(np.unravel_index(np.argmax(a[1:,1:]),a[1:,1:].shape))
print(a[1:,1:][indx])


# Structure for automatic step-size implementation statistics
@dataclass
class stats:
    nsteps: int = 0     # Total number of steps             --- Общее число шагов
    nrej: int = 0       # Number of rejected steps          --- Число отброшенных шагов
    trej: float = 0     # Rejected steps times              --- Абсциссы отброшенных шагов
    hrej: float = 0     # Rejected steps sizes              --- Длины отброшенных шагов
    nfevals: int = 0    # Right-hand sides calculations     --- Количество оценок правйо части (вычислений функции f)


# Constant step Runge--Kutta solution
def rkconst(f, t0: float, y0: float, tfin: float, N: int, method, xi=0):
    """
    f: right-hand side of the equation      --- правая часть диффура (функция)
    t0, y0: initial point                   --- начальная точка
    tfin: end time point                    --- конечная точка по времени
    N: number of constant steps to make     --- число шагов на интервал решения
    method: reference to the method scheme for one step --- ссылка на метод для решения
    xi: the parameter for rk2step function  --- параметр в методе второго порядка
    """

    h = (tfin - t0) / N # step-size         --- длина шага
    d = len(y0)         # equations number  --- число уравнений в системе

    # T --- times in the mesh points        --- абсциссы точек сетки
    # Y --- values in the mesh points       --- решение в точках сетки

    T = np.zeros(N + 1)
    Y = np.zeros((N + 1, d))

    T[0] = t0
    Y[0] = y0

    for i in range(N):
        new_y = method(f, T[i], Y[i], h, xi)
        new_t = T[i] + h
        T[i+1] = new_t
        Y[i+1] = new_y

    return T, Y.transpose()


# Second order RK method step --- Один шаг методом второго порядка
def rk2step(f, t0, y0, h, xi):
    """
    f: right-hand side of the equation      --- правая часть диффура (функция)
    t0, y0: initial point of the step       --- начальная точка шага
    h: step-size                            --- длина шага
    xi: the parameter for rk2step function  --- параметр в методе второго порядка
    """
    """
    xi   | xi
    -----+-------------------
         |1-1/2xi   1/2xi 
    """
    c2 = xi
    a21 = xi

    b2 = 0.5 / c2
    b1 = 1 - b2

    k1 = f(t0, y0)
    k2 = f(t0 + c2*h, y0 + a21*h*k1)

    y1 = y0 + b1*h*k1 + b2*h*k2

    return y1  # value in t0 + h --- решение в t0 + h


# Third order RK method step
def rk3step(f, t0, y0, h, xi):
    """
    f: right-hand side of the equation      --- правая часть диффура (функция)
    t0, y0: initial point of the step       --- начальная точка шага
    h: step-size                            --- длина шага
    xi : NOT USED HERE (added for compatibility with rk2step)
    """

    """
    0   |
    1/3 | 1/3
    2/3 |  0    2/3
    ----+-------------------
        |1/4     0      3/4
    """
    c2 = 1/3
    c3 = 2/3

    a21 = 1/3
    a32 = 2/3

    b1 = 1/4
    b3 = 3/4

    t1 = t0 + c2*h
    t2 = t0 + c3*h

    k1 = f(t0, y0)
    k2 = f(t1, y0 + a21*h*k1)
    k3 = f(t2, y0 + a32*h*k2)

    y1 = y0 + b1*h*k1 + b3*h*k3

    return y1  # value in t0 + h --- решение в t0 + h


# Fourth order RK method step
def rk4step(f, t0, y0, h, xi):
    """
    f: right-hand side of the equation      --- правая часть диффура (функция)
    t0, y0: initial point of the step       --- начальная точка шага
    h: step-size                            --- длина шага
    xi : NOT USED HERE (added for compatibility with rk2step)
    """
    """
    0   |
    1/2 | 1/2
    1/2 |  0    1/2
    1   |  0     0      1
    ----+---------------------------
        |1/6    1/3    1/3      1/6
    """
    c2 = 1 / 2
    c3 = 1 / 2
    c4 = 1

    a21 = 1 / 2
    a32 = 1 / 2
    a43 = 1

    b1 = 1 / 6
    b2 = 1 / 3
    b3 = 1 / 3
    b4 = 1 / 6

    t1 = t0 + c2 * h
    t2 = t0 + c3 * h
    t3 = t0 + c4 * h

    k1 = f(t0, y0)
    k2 = f(t1, y0 + a21 * h * k1)
    k3 = f(t2, y0 + a32 * h * k2)
    k4 = f(t3, y0 + a43 * h * k3)

    y1 = y0 + b1*h*k1 + b2*h*k2 +b3*h*k3 + b4*h*k4

    return y1  # value in t0 + h --- решение в t0 + h


# Embedded RK3(2) method with FSAL: autostep (similar to ode23 in matlab)
def rk32f(f, t0: float, y0, tfin: float, hmax: float, tol: float):
    """
    f: right-hand side of the equation      --- правая часть диффура (функция)
    t0, y0: initial point                   --- начальная точка
    tfin: end time point                    --- конечная точка по времени
    hmax: maximal step-size                 --- максимальная длина шага
    tol: local absolute tolerance           --- локальная погрешность
    """
    st = stats()  # statistics instance      --- статистика по шагам и вычислениям

    # initial step-size
    p = 3
    a = max(np.abs(t0), np.abs(tfin))
    b = np.linalg.norm(f(t0, y0))
    st.nfevals = 1
    delta = np.power(1 / a, p + 1) + np.power(b, p + 1)
    h = np.power(tol / delta, 1 / (p + 1))
    if h > hmax:
        h = hmax

    # T # times in the mesh points          --- абсциссы точек сетки
    # Y # values in the mesh points         --- решение в точках сетки
    T = [t0]
    Y = [y0]
    fac_min = 0.5
    fac_max = 3
    factor = 0.9

    new_y, error = rk32fstep(f, T[-1], Y[-1], h)
    st.nfevals += 3

    while np.linalg.norm(error) > tol:
        st.trej += 1
        h = h / 2
        new_y, error = rk32fstep(f, T[-1], Y[-1], h)
        # !!!!! st.nfevals += 3


    Y.append(new_y)
    T.append(T[-1] + h)

    while T[-1] + h < tfin:

        tmp = np.power(tol / np.linalg.norm(error), 1 / p)
        h = h * max(fac_min, min(fac_max, factor * tmp))
        new_y, error = rk32fstep(f, T[-1], Y[-1], h)
        st.nfevals += 3
        while np.linalg.norm(error) > tol:
            st.trej += 1
            h /= 2
            new_y, error = rk32fstep(f, T[-1], Y[-1], h)
            # !!!!! st.nfevals += 3
        Y.append(new_y)
        T.append(T[-1] + h)

    # last step
    h = tfin - T[-1]
    new_y, error = rk32fstep(f, T[-1], Y[-1], h)
    st.nfevals += 3
    Y.append(new_y)
    T.append(tfin)

    st.nsteps += len(T) - 1

    return np.array(T), np.array(Y).transpose(), st

# Embedded RK3(2) method with FSAL: one step (similar to ode23 in matlab)
def rk32fstep(f, t0, y0, h):
    """
    f: right-hand side of the equation      --- правая часть диффура (функция)
    t0, y0: initial point of the step       --- начальная точка шага
    h: step-size                            --- длина шага
    """
    """
    0   |
    1   |    1
    1/2 |   1/4   1/4
    ----+-------------------
    y2  |   1/2   1/2
    y1  |   1/6   1/6    2/3
    """
    c2 = 1
    c3 = 1/2

    a21 = 1
    a31 = 1/4; a32 = 1/4

    b11 = 1/2; b12 = 1/2
    b21 = 1/6; b22 = 1/6; b23 = 2/3

    t1 = t0 + c2 * h
    t2 = t0 + c3 * h

    k1 = f(t0, y0)
    k2 = f(t1, y0 + a21*h*k1)
    k3 = f(t2, y0 + a31*h*k1 + a32*h*k2)

    # Approximation by the main method of order 3 at t0 + h --- Приближение методом 3-го порядка в t0 + h
    y1 = y0 + b21*h*k1 + b22*h*k2 + b23*h*k3
    # Approximation by the estimator of order 2 at t0 + h --- Приближение методом 2-го порядка в t0 + h
    y2 = y0 + b11*h*k1 + b12*h*k2

    # Error estimation
    err = y1 - y2

    return y1, err
