from animal import Animal
from os import path

class Zoo:
    def __init__(self):
        self.data = self._read_data()

    def _read_data(self):
        zoo_path = path.relpath("datasets/zoo.data")
        zoo_file = open(zoo_path)
        animals = []
        for line in zoo_file:
            animals.append(Animal(line.strip()))
        return animals

    def format(self):
        formatted_data = "Zoo\n\n"
        for animal in self.data:
            formatted_data += animal.format() + "\n"
        return formatted_data
