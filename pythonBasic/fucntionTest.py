def func(a, b):
    c = a + b
    func1(a, b)
    print('the func c is ', c)


def func1(a, b):
    c = a + b
    print('the func1 c is ', c)


func(1, 2)


def sale_car(price, color='red', brand='carmy', is_second_hand=True):
    print('price', price,
          'color', color,
          'brand', brand,
          'is_second_hand', is_second_hand, )


sale_car(12, 'sadsda')


def report(name, *grades):
    total_grade = 0
    for grade in grades:
        total_grade += grade
    print(name, 'total grade is ', total_grade)


report('Mike', 8, 9,20,25)

if __name__ == '__main__':
    print("hhaa")



