from models.zoo import Zoo
import numpy as np


class ZooSample:
    def __init__(self):
        self._zoo = Zoo()
        self.nb_inputs = 16
        self.nb_outputs = 7

    def samples(self):
        sample_size = self.size()
        samples = np.zeros(sample_size, dtype=[('input', int, self.nb_inputs), ('output', int, self.nb_outputs)])

        for i in range(sample_size):
            samples[i] = self._input_to_tuple(self._zoo.data[i]), self._output_to_tuple(self._zoo.data[i])

        return samples

    def _input_to_tuple(self, animal):

        return (animal.hair,
                animal.feathers,
                animal.eggs,
                animal.milk,
                animal.airborne,
                animal.aquatic,
                animal.predator,
                animal.toothed,
                animal.backbone,
                animal.breathes,
                animal.venomous,
                animal.fins,
                animal.legs,
                animal.tail,
                animal.domestic,
                animal.catsize)

    def _output_to_tuple(self, animal):
        output = [0, 0, 0, 0, 0, 0, 0]
        output[animal.type - 1] = 1
        return tuple(output)

    def size(self):
        return len(self._zoo.data)

    def format(self):
        return self._zoo.format()

