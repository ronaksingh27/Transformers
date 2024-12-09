 
class Test():
    def __init__(self,name,age):
        self.name = name
        print("age from init",age)

    def get_age(self):
        print("age from func" , self.age)
       

model = Test("Ronak",22)
model.get_age