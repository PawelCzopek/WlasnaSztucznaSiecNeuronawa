# Sztuczne sieci neuronowe
# Projekt
# Pawel Czopek 136533
# WARiE AIR SSiR

# moduly
import sys # pobranie argumentow wywolania
import json # odczytywanie formaty JSON

# biblioteki
import numpy as np # biblioteka matematyczna

import tensorflow as tf
from tensorflow import keras

# Zmienne glogalne
xtrain = [] # dane wejsciowe do trenowania
ytrain = [] # dane do wyjsciowe do trenowania
xtest = [] # dane wejsciowe do testowania
ytest = [] # dane wyjsciowe do testowania

# Zadanie 5
# funkcje przejsc
def ReLU(x):
    if x > 0.0:
        y = x
    else:
        y = 0.0
    return y

def tanh(x):
    y = np.tanh(x)
    return y

def ELU(x):
    a = 1.0
    if x > 0.0:
        y = x
    else:
        y = a*(np.exp(x)-1)
    return y

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

# Zadanie 6
# funkcje odwortne
def odwReLU(x):
    if x <= 0:
        y = 0
    elif x > 0:
        y = 1
    return y

def odwtanh(x):
    y = 1 - ((tanh(x))**2)
    return y

def odwELU(x):
    a=1
    if x >= 0:
        1
    else:
        a*np.exp(x)
    return y

def odwsigmoid(x):
    y = sigmoid(x)*(1-sigmoid(x))
    return y

# Zadanie 11
# wczytanie danych
def readDataFile(filename):
    global xtrain, ytrain, xtest, ytest
    f = open(filename, "r", encoding="utf-8")
    dataLines = f.readlines()
    f.close()
    # podzial dzanych
    lenght = len(dataLines)
    stop = lenght*2/3
    for d in dataLines:
        idT, xT, yT = d.split(" ")
        idT, xT, yT = float(idT), float(xT), float(yT)
        #print(idT)
        if idT<stop:
            xtrain.append(xT)
            ytrain.append(yT)
        else:
            xtest.append(xT)
            ytest.append(yT)
    #print(str(xtrain[1])+" "+str(ytrain[1]))
    #print(str(xtest[1])+" "+str(ytest[1]))
    xtrain = np.array(xtrain, dtype='float')
    ytrain = np.array(ytrain, dtype='float')
    xtest = np.array(xtest, dtype='float')
    ytest = np.array(ytest, dtype='float')
    # Zadanie 12
    # preprocesing danych
    full_datax = np.concatenate((xtrain, xtest))
    max_datax = max(full_datax)
    full_datay = np.concatenate((ytrain, ytest))
    max_datay = max(full_datay)
    xtrain, xtest = xtrain / max_datax, xtest / max_datax
    ytrain, ytest = ytrain / max_datay, ytest / max_datay

# Zadanie 13
# odczyt pliku JSON
def readJsonFile(filename):
    f_json = open(filename, "r", encoding="utf-8")
    json_data = json.load(f_json)
    f_json.close()
    #print(json_data)
    if json_data["reg"] != "L1" and json_data["reg"] != "L2":
        print("Bledna wartosc reg")
    if json_data["loss"] != "hinge" and json_data["loss"] != "softmax":
        print("Bledna wartosc loss")

    return json_data

# Klasy
# Zadanie 7
class neuron:

    def __init__(self, input_num=1, act_fun='ReLU'):
        self.input_num = input_num
        self.act_fun = act_fun
        self.w = np.ones((1,self.input_num), dtype=float) # wagi
        self.g = np.ones((1,self.input_num), dtype=float) # gradient
        self.input = np.ones((1,self.input_num), dtype=float)

    def forward(self, input):
        out = np.sum(self.w * input)
        if self.act_fun == 'ReLU':
            out = ReLU(out)
        elif self.act_fun == 'tanh':
            out = tanh(out)
        elif self.act_fun == 'ELU':
            out = ELU(out)
        elif self.act_fun == 'sigmoid':
            out = sigmoid(out)
        else:
            print('Bledna funkcja aktywacji')

        self.input = input
        return out

    def backward(self, dout):
        g=self.g
        if self.act_fun == 'ReLU':
            dz = odwReLU(dout)
        elif self.act_fun == 'tanh':
            dz = odwtanh(dout)
        elif self.act_fun == 'ELU':
            dz = odwELU(dout)
        elif self.act_fun == 'sigmoid':
            dz = odwsigmoid(dout)
        else:
            print('Bledna funkcja aktywacji')

        for k in g:
            k = dz

        self.g = g * self.input
        dinput = g * self.w

        return dinput

# Zadanie 8
class warstwa:
    # Zmienne klasy
    neurons = []

    def __init__(self, neuron_num=1, input_num=1, act_fun='ReLU'):
        self.neuron_num = neuron_num
        for i in range(self.neuron_num):
            self.neurons.append(neuron(input_num, act_fun))

    # funkcja obliczajaca wyjscie warstwy
    def forwardW(self, input):
        outputW = np.zeros(self.neuron_num)
        for i in range(0,self.neuron_num):
            #print(x)
            outputW[i] = self.neurons[i].forward(input)
        return outputW

    # funkcja obliczajaca gradienty
    def backwardW(self, input):
        for i in range(0,self.neuron_num):
            neurons[i].backward(input[i])
        return out_gradient

# Zadanie 14
class siec:
    # Zmienne klasy
    layers = []

    def __init__(self, layer):
        self.layer_num = len(layer)
        for i in range(0, self.layer_num):
            if i == 0:
                self.layers.append(warstwa(layer[i]['liczba_neuronów'], 1, layer[i]['funkcja_aktywacji']))
            else:
                self.layers.append(warstwa(layer[i]['liczba_neuronów'], layer[i-1]['liczba_neuronów'], layer[i]['funkcja_aktywacji']))

    # Metoda obliczajaca wyjscie klasy
    def forwardS(self, input):
        # rzutowanie liczb na typ tablic numpy
        if type(input) == type(0) or type(input) == type(0.0):
            outputT = np.array([input])
        else:
            outputT = input
        # sprawdzenie poprawnosci wejscia
        if len(outputT) != self.layers[0].neuron_num:
            print("Liczba dabnych wejsciowych nie zgadza sie liczba wejsc sieci")
        else:
            for i in range(0, self.layer_num):
                outputT = self.layers[i].forwardW(outputT)

        output = outputT
        return output

# __________________________________________________________________________
# main

# Zadanie 4
# sprawdzenie poprawnej liczby argumentow
if len(sys.argv) != 3:
    print('Nieodpowiednia liczba argumentow wywolania')
    exit()

# pobranie argumentow wejsciowych
filename_json = sys.argv[1]
filename_data = sys.argv[2]

# test funkcji aktywacji
print(ReLU(1))
print(tanh(1))
print(sigmoid(1))
print(ELU(1))
print('OK')

# test klasy neuron
neur = neuron(3)
neur.__init__(2)
print('Test neurona')
print('Wagi:')
print(neur.w)

n1 = neuron(5, 'ReLU')
print(n1.forward(np.array([1, 2, 3, 4, 5])))
print(n1.backward(15))
print('Gradient:')
print(n1.g)

# Zadanie 9
# inicjalizacja sieci;
warstwaWe = warstwa(1, 1)
warstwaUkr = warstwa(10, 1)
warstwaWy = warstwa(1, 10)

out1 = warstwaWe.forwardW(5)
out2 = warstwaUkr.forwardW(out1)
out3 = warstwaWy.forwardW(out2)

print('___________________________')
print('Zadanie 9')
print('"Moja" siec')
print(out3)

# Zadanie 10
# siec referencyjna
TF = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1,1)),
  #tf.keras.layers.Dense(units=1, input_shape=[1,1], activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)#, activation='relu')
])

TF2 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1,1)),
  #tf.keras.layers.Dense(units=1, input_shape=[1,1], activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(1)#, activation='relu')
])

TF3 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1,1)),
  #tf.keras.layers.Dense(units=1, input_shape=[1,1], activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(1)#, activation='relu')
])

# wywolanei funkcji zad 11, 12
readDataFile(filename_data)
print('___________________________')
print('Zadanie 12')
print('Dane do nauki')
print(len(xtrain))
print(ytrain)
print(type(xtrain))
print(type(xtrain[1]))

# wywalanie funkcji zad 13
learn_param = readJsonFile(filename_json)
# test zad 13
lr = learn_param["lr"]
reg = learn_param["reg"]
loss = learn_param["loss"]
warstwy = learn_param["warstwy"]
print('___________________________')
print('Zadzanie 13: JSON')
print(learn_param)
print(lr)
for x in warstwy:
    w1fakt = x["funkcja_aktywacji"]
    w1numneu = x["liczba_neuronów"]
    print(w1fakt)
    print(w1numneu)

# test sieci - zad 14
modelTF = siec(warstwy)
modelRAW = siec(warstwy)
print('___________________________')
print('Zadanie  14')
print(modelTF.forwardS(5))

# Zadanie 15
# uczenie modelu tensorflow
loss_fn = tf.keras.losses.Hinge()
#TF.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.CategoricalAccuracy()])
TF.compile(optimizer='Adagrad', loss=loss_fn, metrics=['accuracy'])
print('___________________________')
TF.fit(xtrain, ytrain, epochs=30)
TF.evaluate(xtest,  ytest, verbose=2)
print('___________________________')
print(TF(xtest[:5]))
print(ytest[:5])

# wypisanie wag z modelu sieci tensorflow
#TF.summary()
wagi_tf = TF.weights
wagi_tf = wagi_tf
print('___________________________')
for x in wagi_tf:
    print(x)
    print('_________')
