def calc(expr):
    """ вычисляет значение выражения, записанного в обратной польской нотации"""
    stack = []
    polish_not = polish_notation(expr)
    for element in polish_not:
        if isinstance(element, int):
            stack.append(element)
        else:
            second = stack.pop()
            first = stack.pop()
            if element == '*':
                stack.append(first * second)
            elif element == '/':
                stack.append(first / second)
            elif element == '+':
                stack.append(first + second)
            elif element == '-':
                stack.append(first - second)

    return stack


def polish_notation(expr):
    """возвращает обратную польскую нотацию в виде списка"""
    polish_not = []
    stack = []
    operations = {'*', '/', '+', '-'}

    # словарь операция:приоритет
    dictionary = {'*': 2, '/': 2, '+': 1, '-': 1, '(': 0, ')': 0}

    digits = [str(x) for x in range(10)]

    # переменные для чтения чисел
    number = 0
    is_reading_number = False

    for symbol in expr:
        # читаем или продолжаем читать число
        if symbol in digits:
            is_reading_number = True
            number = int(symbol) + number * 10

        else:
            # если завершили читать число, положить его в выходной список
            if is_reading_number:
                is_reading_number = False
                polish_not.append(number)
                number = 0
            # если встретилась открывающая скобка, положить в стак
            if symbol == "(":
                stack.append(symbol)
            # если встретилась закрывающая скобка, то вытолкнуть все элементы до открывающей
            # скобки из стака в выходный список; открывающую скобку выбросить из стака
            elif symbol == ")":
                elem = stack[-1]
                while elem != '(':
                    polish_not.append(stack.pop())
                    elem = stack[-1]
                stack.pop()

            elif symbol in operations:
                priority_of_symbol = dictionary[symbol]
                if not stack:
                    priority_of_symbol_in_stack = 0
                else:
                    priority_of_symbol_in_stack = dictionary[stack[-1]]
                # пока приоритет операции из стака больше либо равен приоритету встретившейся операции,
                # выталкиваем всё из стака в выходной список
                while priority_of_symbol <= priority_of_symbol_in_stack:
                    # выталкиваем всё в выход
                    polish_not.append(stack.pop())
                    if not stack:
                        priority_of_symbol_in_stack = 0
                    else:
                        priority_of_symbol_in_stack = dictionary[stack[-1]]
                stack.append(symbol)
    # если после завершения алгоритма осталось в памяти число (это происходит когда число
    # стоит в конце вычисляемого выражения), поместить его в выходной список
    if is_reading_number:
        polish_not.append(number)
    # вытолкнуть всё из стака в выходной список
    while stack:
        polish_not.append(stack.pop())
    return polish_not


print("answer is {}".format(calc('2*(15-3*4)-2')), 'expected 4')
print("answer is {}".format(calc('2*(15+3-5)')), 'expected 26')
print("answer is {}".format(calc('(48*3+56)/7')), 'expected 28.57')
print("answer is {}".format(calc('(123/2+1*3*(123-3))*1')), 'expected 421.5')


