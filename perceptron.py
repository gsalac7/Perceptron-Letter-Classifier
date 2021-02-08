import numpy as np
from dataset import *

def show(image):
    col = 0
    for char in image:
        col += 1
        if char == 1:
            print("#", end='')
        else:
            print(" ", end='')
        if col == 5:
            col = 0
            print("")
    print("", end='')


show(TRAINING_SET[0][0])
show(TRAINING_SET[1][0])
show(TRAINING_SET[2][0])
show(TRAINING_SET[3][0])
show(TRAINING_SET[4][0])
class Perceptron:
    def __init__(self, letter):
        self.weights = np.random.rand(35)
        self.learning_rate = 0.01
        self.bias = -5
        self.letter= letter
    

    # Add two vectors together
    def add_list(self, list1, list2):
        result = []
        for i in range(len(list1)):
            result.append(list1[i] + list2[i])
        return result

    # product of two vectors
    def product(self, multiple, inputs):
        product = []
        for i in inputs:
            product.append(i * multiple)
        return product

    # Update function taken from 1.4 from textbook
    # weights = weights + learning_rate(actual_value - predicted_value) * inputs
    def update_function(self, predicted_value, inputs):
        new_weights = self.product(self.learning_rate*(self.actual_value - predicted_value), inputs)
        new_weights = self.add_list(self.weights, new_weights)
        return new_weights
        
    # Activation Function
    def sign(self, val):
        if val > 0:
            return 1 # It is the letter A
        else:
            return -1 # It is NOT the letter A

    # Run the guess function
    # Sum of all inputs * weights
    def guess(self, inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        sum += self.bias
        return self.sign(sum)

    # Error function
    def error(self, actual_value, predicted_value):
        err = actual_value - predicted_value
        return err

    # run the algorithm to classify a letter
    def learn(self, inputs):
        EPOCH = 1
        self.actual_value = 1
        self.predicted_value = 0 
        self.errors_exist = True
        while self.errors_exist == True:
            print("EPOCH " + str(EPOCH) + " to train for letter: " + self.letter)
            print("*****")
            print("Starting FONT 1 TRAINING: ")
            print("*****")
            EPOCH += 1
            for i in range(len(inputs)):
                self.predicted_value = self.guess(inputs[i][0])
                if inputs[i][1]== self.letter:
                    self.actual_value = 1
                else:
                    self.actual_value = -1
                # the guess is incorrect, need to fix
                if self.actual_value != self.predicted_value:
                    self.weights = self.update_function(self.predicted_value, TRAINING_SET[i][0])
                    self.errors_exist = True

                #There is no error for this iteration
                else:
                    self.errors_exist = False

                if i == 26:
                    print("*****")
                    print("Starting FONT 2 TRAINING: ")
                    print("*****")
                print("Predicted Value: " + str(self.predicted_value) + "   " + str(TRAINING_SET[i][1]) + " Actual Value: " + str(self.actual_value) + "   " + str(TRAINING_SET[i][1]))

# Run the perceptron to classify the letter "A"          
def main():
    pA = Perceptron('A')
    pA.learn(TRAINING_SET)

    '''
    Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','J','K','L','M', 'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    perceptron_obj = {} 
    for i in range(len(Alphabet)):
        perceptron_obj[Alphabet[i]] = Perceptron(Alphabet[i])
        perceptron_obj[Alphabet[i]].learn(TRAINING_SET)
        
        '''

if __name__ == "__main__":
    main()
