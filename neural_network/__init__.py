from numpy import exp, dot, save

# Модель Перцептрона
class NeuralNetwork():
    def __init__(self, weights):
        self.synaptic_weights = weights
            
    # Сигмоидальная функция, описывающая кривую в форме буквы "S".
    # Передаем взвешенную сумму входов через эту функцию для
    # нормализации их в диапазоне от 0 до 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Производная сигмоидальной функции.
    # Это градиент кривой сигмоида.
    # Он показывает, насколько уверены мы в существующем весе.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
        
    # Сохранить корректированные веса
    def save_adjusted_weights(self):
        save("perceptron.npy", self.synaptic_weights)
        
    # Мы обучаем нейронную сеть методом проб и ошибок.
    # Каждый раз корректируем веса синапсов.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Передаем обучающий набор через нашу нейронную сеть (один нейрон).
            output = self.think(training_set_inputs)

            # Вычисляем ошибку (разницу между желаемым результатом
            # и предсказанным результатом).
            error = training_set_outputs - output

            # Умножаем ошибку на вход и еще раз на градиент кривой сигмоида.
            # Это означает, что менее уверенные веса корректируются больше.
            # Это также означает, что входы, которые равны нулю, не вызывают изменения весов.
            adjustment = dot(training_set_inputs.T, error * (self.__sigmoid_derivative(output)))

            # Корректируем веса.
            self.synaptic_weights += adjustment

    # Нейронная сеть "думает"
    def think(self, inputs):
        # Передаем входы через нашу нейронную сеть (один нейрон).
        s = dot(inputs, self.synaptic_weights)
        return self.__sigmoid(s)