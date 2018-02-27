#from numpy import *
from random import *
from math import *
import numpy as np

class Connection():
	def __init__(self):
		self.weight = 0
		self.deltaWeight = 0
		return

class Layer():
	def __init__(self):
		return

class Neuron():
	def __init__(self, numOutputs, myIndex):
		self.outputVal = 0
		self.outputWeights = []

		self.myIndex = myIndex
		self.gradient = 0

		self.eta = 0.15# [0.0..1.0]
		self.alpha = 0.5# [0.0..n]

		for c in range(0, numOutputs):
			self.outputWeights.append(Connection())
			self.outputWeights[-1].weight = self.randomWeight()
		return

	def randomWeight(self):
		return random()

	def sumDOW(self, nextLayer):

		sum = 0.0
		# sum our contributions of the errors at the nodes we feed (not including bias)
		for n in range(0, len(nextLayer)-1):
			sum += self.outputWeights[n].weight * nextLayer[n].gradient

		return sum

	def activationFunction(self, x):
		# tanh
		return np.tanh(x)

	def activationFunctionDerivative(self, x):
		return 1.0 - x*x
		#return np.tanh(x)**2

	def getOutputValue(self):
		return self.outputVal

	def setOutputValue(self, val):
		self.outputVal = val
		return


	def calcOutputGradients(self, targetVal):
		delta = targetVal - self.outputVal
		self.gradient = delta * self.activationFunctionDerivative(self.outputVal)

	def calcHiddenGradients(self, nextLayer):
		dow = self.sumDOW(nextLayer)
		self.gradient = dow * self.activationFunctionDerivative(self.outputVal)

	def updateInputWeights(self, prevLayer):
		
		# the weights to be updated are in the connection container
		# in the neuron in the preceding layer
		# don't include bias
		#for n in range(0, len(prevLayer)-1):
		for n in range(0, len(prevLayer)):
			neuron = prevLayer[n]
			#print("weights: " + str(neuron.outputWeights))
			#print(self.myIndex)
			

			# complete hack:
			# if(self.myIndex > len(neuron.outputWeights)-1):
			# 	self.myIndex = len(neuron.outputWeights)-1

			oldDeltaWeight = neuron.outputWeights[self.myIndex].deltaWeight

			# newDeltaWeight =
			# # individual input, magnified by the gradient and train rate:
			# self.eta
			# * neuron.outputVal
			# * self.gradient
			# # also add momentum = a fraction of the previous delta weight
			# + self.alpha
			# *oldDeltaWeight

			newDeltaWeight = self.eta * neuron.outputVal * self.gradient + self.alpha * oldDeltaWeight

			neuron.outputWeights[self.myIndex].deltaWeight = newDeltaWeight
			neuron.outputWeights[self.myIndex].weight += newDeltaWeight
		return prevLayer

	def feedForward(self, prevLayer):
		
		# sum the previous layer's outputs (which are our inputs)
		# include the bias node from the previous layer

		sum = 0.0

		for n in range(0, len(prevLayer)):
			sum += (prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[self.myIndex].weight)

		self.outputVal = self.activationFunction(sum)


class Net():
	def __init__(self, topology):

		self.layers = []
		self.error = 0.0
		self.recentAverageError = 0.0
		self.recentAverageSmoothingFactor = 100.0

		numLayers = len(topology)

		for layerNum in range(0, numLayers):
			# self.layers.append(Layer())
			self.layers.append([])

			numOutputs = 0
			# if it's the last layer the number of outputs is 0
			if(layerNum == numLayers-1):
				numOutputs = 0
			else:
				# otherwise it's the number of neurons in the next layer
				numOutputs = topology[layerNum+1]

			# adds an extra neuron for bias:
			for neuronNum in range(0, topology[layerNum]+1):
				self.layers[-1].append(Neuron(numOutputs, neuronNum))
				#print("Made a Neuron!")

			# force the bias node's output value to 1.0
			# it's the last neuron created above
			#if(len(self.layers) > 0):
			#self.layers[-1][-1].outputVal = 1.0
			self.layers[-1][-1].setOutputValue(1.0)

	def feedForward(self, inputVals):
		if(len(inputVals) != len(self.layers[0])-1):
			raise ValueError('input layer is not the same size as it should be.')

		# assign the input values to the input neurons
		for i in range(0, len(inputVals)):
			self.layers[0][i].setOutputValue(inputVals[i])



		# forward propagate aka find our answer:
		# skip input layer:
		# for each layer:
		#for layerNum in range(1, len(self.layers)+1):
		for layerNum in range(1, len(self.layers)):
			prevLayer = self.layers[layerNum-1]

			#print(prevLayer)

			# for each neuron:
			# don't include bias neuron (-1 to loop)
			for n in range(0, len(self.layers[layerNum])-1):
				self.layers[layerNum][n].feedForward(prevLayer)


		return


	def backProp(self, targetVals):

		# calculate overall net error (root mean square error)
		outputLayer = self.layers[-1]
		self.error = 0.0

		# for each neuron in the output layer (not including bias)
		for n in range(0, len(outputLayer)-1):
			delta = targetVals[n] - outputLayer[n].outputVal
			self.error += delta * delta
		# get the average
		self.error /= len(outputLayer) - 1
		self.error = sqrt(self.error)

		# recent average measurement:
		self.recentAverageError = (self.recentAverageError * self.recentAverageSmoothingFactor + self.error) / (self.recentAverageSmoothingFactor + 1.0)


		# calculate output layer gradients
		# for each neuron in the output layer (not including bias)
		for n in range(0, len(outputLayer)-1):
			outputLayer[n].calcOutputGradients(targetVals[n])






		# calculate hidden layer gradients
		# start from layerSize-2 down to 0
		#for layerNum in range(len(self.layers)-2, 0, -1):
		for layerNum in range(len(self.layers)-2, 0, -1):
			hiddenLayer = self.layers[layerNum]
			nextLayer = self.layers[layerNum+1]

			# loop through neurons in hidden layer:
			for n in range(0, len(hiddenLayer)):
				#hiddenLayer[n].calcHiddenGradients(nextLayer)
				self.layers[layerNum][n].calcHiddenGradients(nextLayer)





		# for all layers from outputs to first hidden layer
		# update connection weights
		for layerNum in range(len(self.layers)-1, 1, -1):
			layer = self.layers[layerNum]
			prevLayer = self.layers[layerNum-1]

			# for each neuron:
			for n in range(0, len(layer)-1):
				self.layers[layerNum-1] = layer[n].updateInputWeights(prevLayer)



		return

	def getResults(self, resultVals):

		del resultVals[:]

		l = len(self.layers[-1])-1

		for n in range(0, l):
			val = self.layers[-1][n].outputVal
			resultVals.append(val)

		return







def main():

	topology = [3, 3, 3, 1]
	myNet = Net(topology)

	for i in range(0, 40000):

		a = randint(0,1)
		b = randint(0,1)
		c = randint(0,1)
		# a = random()
		# b = random()
		# c = random()
		ans = a+b+c
		if(a == b):
			ans = 0

		inputVals = [a, b, c]
		myNet.feedForward(inputVals)

		resultVals = []
		myNet.getResults(resultVals)

		targetVals = [ans]
		myNet.backProp(targetVals)



		#print("pass: " + str(i))
		#print("target: " + str(targetVals[0]))
		#print("output: " + str(resultVals[0]))
		#print("error: " + str(myNet.error))
		print("recent percent error: " + str(myNet.recentAverageError))


main()