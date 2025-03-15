class Fruit:
    def __init__(self, value):
        self.value = value

def my_print(self):
    print(self.value)

'''
apple = Fruit(10)

# apple 객체의 my_print함수로 static의 my_print함수를 넣어주기 위해서 
apple.my_print = my_print를 할 수도 있다.

하지만, self 인자를 넣어줘야 하기 때문에 위에처럼은 못하고
my_print(apple)을 해주면, print 결과물이 apple.my_print에 저장된다.

따라서 my_print method의 descriptor에 apple을 넣어준 함수를 apple.my_print로 설정해줌으로써
apple의 my_print로 탄생하게 됨
'''

apple = Fruit(10)
apple.my_print = my_print(apple)
apple.my_print()
#print(my_print(apple))