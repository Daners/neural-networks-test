class MyClass:
    Greeting =''

    def __init__(self, Name='there'):
        self.Greeting = Name + '!'

    def SayHello(self):
        print("Hello {0}".format(self.Greeting))

myclass = MyClass("Daniel")
print myclass.SayHello()


class Car(object):
	"""
		blueprint for car
	"""

	def __init__(self, model, color, company, speed_limit):
		self.color = color
		self.company = company
		self.speed_limit = speed_limit
		self.model = model

	def start(self):
		print("started")

	def stop(self):
		print("stopped")

	def accelarate(self):
		print("accelarating...")
		"accelarator functionality here"

	def change_gear(self, gear_type):
		print("gear changed")
		" gear related functionality here"

maruthi_suzuki = Car("ertiga", "black", "suzuki", 60)
print maruthi_suzuki.color
