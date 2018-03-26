if __name__ == '__main__':
    print('Hello world' + ' Hello Hong Kong')
    print(int('2') + 3)
    print(float('1.2') + 3)
    print(3 ** 3)

    apple = 1  # 赋值 数字
    print(apple)

    apple = 'iphone 7 plus'  # 赋值 字符串
    print(apple)

a, b, c = 11, 12, 13
print(a, b, c)

# condition = 0
# while condition < 10:
#     print(condition)
#     condition = condition + 1
#

a = range(10)
while a:
    print(a[-1])
    a = a[:len(a) - 1]

example_list = [1, 2, 3, 4, 5, 6, 7, 12, 543, 876, 12, 3, 2, 5]
for i in example_list:
    print(i)

example_list = [1, 2, 3, 4, 5, 6, 7, 12, 543, 876, 12, 3, 2, 5]
for i in example_list:
    print(i)
    print('inner of for')
print('outer of for')

# step 代表的为步长，即相隔的两个值得差值。从 start 开始，依次增加 step 的值，直至等于或者大于 stop
for i in range(0, 13, 5):
    print(i)

tup = ('python', 2.7, 64)
for i in tup:
    print(i)

dic = {}
dic['lan'] = 'python'
dic['ve rsion'] = 2.7
dic['platform'] = 64
for key in dic:
    print(key, dic[key])

s = set(['python', 'python2', 'python3', 'python'])
for item in s:
    print(item)

def abc():
    print('This is a function')
    a = 1+2
    print(a)

    abc()

# define a Fib class
class Fib(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()


# using Fib object
for i in Fib(5):
    print(i)


    def fib(max):
        a, b = 0, 1
        while max:
            r = b
            a, b = b, a + b
            max -= 1
            yield r


    # using generator
    for i in fib(5):
        print(i)

    x = 1
    y = 2
    z = 3
    if x < y or x < z:
        print('x is less than y')

x = 4
y = 2
z = 3
if x > y:
    print('x is greater than y')
else:
    print('x is less or equal y')

    worked = True
    result = 'done' if worked else 'not yet'
    print(result)
x = 4
y = 2
z = 3
if x > 1:
    print ('x > 1')
elif x < 1:
    print('x < 1')
else:
    print('x = 1')
print('finish')


