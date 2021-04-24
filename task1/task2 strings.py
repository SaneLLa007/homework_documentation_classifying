def is_empty(dict):
    """Возвращает True если все значения словаря равны нулю"""
    for key in dict.keys():
        if dict[key] != 0:
            return False
    return True


def check_inv(a: str, b: str):
    dictionary = dict()

    # словарь типа символы : их кол-во в строке b, так мы собрали все доступные символы
    for symbol in b:
        if symbol in dictionary.keys():
            dictionary[symbol] += 1
        else:
            dictionary[symbol] = 1

    dictionary_tmp = dictionary.copy()

    for symbol in a:
        # если встречается символ из словаря, то уменьшаем доступное кол-во такого символа
        # если словарь пустой, значит нашлась такая перестановка
        # если цепочка прерывается (встретили символ не из словаря), словарь становится снова исходным (полным)
        if symbol in dictionary_tmp.keys():
            if dictionary_tmp[symbol] > 0:
                dictionary_tmp[symbol] -= 1
                if is_empty(dictionary_tmp):
                    return True

        else:
            dictionary_tmp = dictionary.copy()

    return False

print(check_inv('abcrotm', 'tro'))
print(check_inv("abqwertOcrgyfqwertyftdfotm", "qwerty"))
print(check_inv("abqwertOcrgyfqwertyftdfotm", "rqewty"))
print(check_inv("abqwertOcrgyfqwertOftdfotm", "qwerty"))
