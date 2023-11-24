from neural_network import NeuralNetwork
from numpy import array, random, load
import os

if __name__ == '__main__':
    
    weights = None
    
    if not os.path.exists("perceptron.npy"):
        # Задаем начальное значение генератора случайных чисел, чтобы получать одни и те же числа
        # при каждом запуске программы.
        random.seed(1)
        
        # Мы моделируем один нейрон с 3 входами и 1 выходом.
        # Присваиваем случайные веса матрице 3 x 1 со значениями в диапазоне от -1 до 1
        # и средним значением 0.
        weights = 2 * random.random((3, 1)) - 1
    else:
        # Загружаем уже корректированные веса с файла "perceptron.npy"
        weights = load("perceptron.npy")
        
    # Инициализируем однонейронную нейронную сеть
    neural_network = NeuralNetwork(weights)

    
    if not os.path.exists("perceptron.npy"):
        print("Случайные начальные веса синапсов:")
    else:
        print("Коррективанные веса синапсов:")
    print(neural_network.synaptic_weights)

    # Обучающий набор. У нас есть 4 примера, каждый состоит из 3 входных значений
    # и 1 выходного значения.
    training_set_inputs = array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])
    
    # [[0],[1],[1],[0]]
    training_set_outputs = array([
        [0, 1, 1, 0]
    ]).T
    
    if not os.path.exists("perceptron.npy"):
        # Обучаем нейронную сеть с использованием обучающего набора
        # Делаем это 10 000 раз и вносим небольшие изменения каждый раз
        neural_network.train(training_set_inputs, training_set_outputs, 10000)
        neural_network.save_adjusted_weights()
        
        print("Новые веса синапсов после обучения:")
        print(neural_network.synaptic_weights)
        
        

    # Проверяем нейронную сеть на новой ситуации
    print("Рассмотрим новую ситуацию [1, 0, 0] -> ?:")
    print(neural_network.think(array([[1, 0, 0]])))
