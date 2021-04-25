class Tree:

    def __init__(self, value=None, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value

    def __iter__(self):
        iterator = []
        iterator.extend(self.parse())
        return iter(iterator)

    def __str__(self):
        answer = ""

        if isinstance(self.value, Tree):
            answer += "vertex is Tree with value {} \n".format(self.value)
        if isinstance(self.value, int):
            answer += "vertex is Int with value {} \n".format(self.value)

        if isinstance(self.left, Tree):
            answer += "left is Tree with value {} \n".format(self.left.value)
        if isinstance(self.left, int):
            answer += "left is Int with value {} \n".format(self.left)

        if isinstance(self.right, Tree):
            answer += "right is Tree with value {}".format(self.right.value)
        if isinstance(self.right, int):
            answer += "right is Int with value {}".format(self.right)

        return answer

    def __repr__(self):
        return str(self.represent())

    def parse(self):
        """возвращает список из листьев дерева(вершины) self"""
        lst = []

        # если попали в лист, возвращаем значение этого листа
        if self.left is None and self.right is None:
            return [self.value]

        # парсим левую вершину
        # если попали в дерево, парсим его, если в лист - возвращаем его значение
        if isinstance(self.left, Tree):
            lst.extend(self.left.parse())
        elif isinstance(self.left, int):
            lst.append(self.left)

        # парсим правую вершину
        if isinstance(self.right, Tree):
            lst.extend(self.right.parse())
        elif isinstance(self.right, int):
            lst.append(self.right)

        return lst

    def represent(self):
        spis = []
        spis.append(self.value)

        if isinstance(self.left, Tree):
            spis.append([self.left.represent()])
        else:
            spis.append([self.left])

        if isinstance(self.right, Tree):
            spis.append([self.right.represent()])
        else:
            spis.append([self.right])
        return spis


tree = Tree(0, Tree(1, Tree(3), Tree(4)),
            Tree(2))
print("\ntree")
print('representation', repr(tree))
print('string:\n', str(tree))
list_iterator = []
for i in tree:
    list_iterator.append(i)
print('iterator', list_iterator)

tree1 = Tree(0, Tree(1, Tree(3, 4, 2), Tree(4, Tree(24, 213, 12))),
             Tree(2))
print("\ntree1")
print('representation', repr(tree1))
print('string:\n', str(tree1))
list_iterator = []
for i in tree1:
    list_iterator.append(i)
print('iterator', list_iterator)

tree2 = Tree(0, 5, 5)
print("\ntree2")
print('representation', repr(tree2))
print('string:\n', str(tree2))
list_iterator = []
for i in tree2:
    list_iterator.append(i)
print('iterator', list_iterator)
