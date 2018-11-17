class Animal:
    def __init__(self, raw_data):
        data = raw_data.split(',')
        self.animal_name = self.convert(data[0], "string")
        self.hair = self.convert(data[1], "bool")
        self.feathers = self.convert(data[2], "bool")
        self.eggs = self.convert(data[3], "bool")
        self.milk = self.convert(data[4], "bool")
        self.airborne = self.convert(data[5], "bool")
        self.aquatic = self.convert(data[6], "bool")
        self.predator = self.convert(data[7], "bool")
        self.toothed = self.convert(data[8], "bool")
        self.backbone = self.convert(data[9], "bool")
        self.breathes = self.convert(data[10], "bool")
        self.venomous = self.convert(data[11], "bool")
        self.fins = self.convert(data[12], "bool")
        self.legs = self.convert(data[13], "int")
        self.tail = self.convert(data[14], "bool")
        self.domestic = self.convert(data[15], "bool")
        self.catsize = self.convert(data[16], "bool")
        self.type = self.convert(data[17], "int")

    def convert(self, data, data_type):
        converted_data = data
        if data_type == "int" or data_type == "bool":
            converted_data = int(data)
        return converted_data

    def format(self):
        formatted_data = ""
        formatted_data += self.formatField("Name", self.animal_name)
        formatted_data += self.formatField("Hair", self.hair)
        formatted_data += self.formatField("Feathers", self.feathers)
        formatted_data += self.formatField("Eggs", self.eggs)
        formatted_data += self.formatField("Milk", self.milk)
        formatted_data += self.formatField("Airborne", self.airborne)
        formatted_data += self.formatField("Aquatic", self.aquatic)
        formatted_data += self.formatField("Predator", self.predator)
        formatted_data += self.formatField("Toothed", self.toothed)
        formatted_data += self.formatField("Backbone", self.backbone)
        formatted_data += self.formatField("Breathes", self.breathes)
        formatted_data += self.formatField("Venomous", self.venomous)
        formatted_data += self.formatField("Fins", self.fins)
        formatted_data += self.formatField("Legs", self.legs)
        formatted_data += self.formatField("Tail", self.tail)
        formatted_data += self.formatField("Domestic", self.domestic)
        formatted_data += self.formatField("Catsize", self.catsize)
        formatted_data += self.formatField("Type", self.type)
        return formatted_data

    def formatField(self, data_name, data_value):
        return "" + data_name + " : {}\n".format(data_value)