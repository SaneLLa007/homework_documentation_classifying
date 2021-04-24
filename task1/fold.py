def foldl(f, x0, lst):
    if not lst:
        return x0
    return foldl(f, f(x0, lst[0]), lst[1:])


def foldr(f, x0, lst):
    if not lst:
        return x0
    return f(lst[0], foldr(f, x0, lst[1:]))


def foldr2(f, x0, lst):
    return foldl(lambda y, x: lambda func: y(f(x, func)), lambda x: x, lst)(x0)


def foldl2(f, x0, lst):
    return foldr(lambda x, y: lambda func: y(f(func, x)), lambda x: x, lst)(x0)


print('left', foldl(lambda x, y: x + y, 0, [1, 2, 3]))
print('left2', foldl2(lambda x, y: x + y, 0, [1, 2, 3]))
print('right', foldr(lambda x, y: x + y, 0, [1, 2, 3]))
print('right2', foldr2(lambda x, y: x + y, 0, [1, 2, 3]))

f = lambda x, y: x / y
print('left', foldl(f, 1, [1, 2, 3]))
print('left2', foldl2(f, 1, [1, 2, 3]))
print('right', foldr(f, 1, [1, 2, 3]))
print('right2', foldr2(f, 1, [1, 2, 3]))
