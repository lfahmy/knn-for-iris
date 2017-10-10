from pandas import *
from math import sqrt

class Knn:
	labeled = DataFrame() #holds data with classes for "training"
	unlabeled = DataFrame() #holds data without classes for testing
	classes = [] #the list of all possible classes
	classFrame = DataFrame() #frame of distance to the test point and corresponding classes
	classDict = {} #keeps track of frequency of each class
	K = 10 #THIS IS A CONSTANT representing k

	def __init__(self, labeled, unlabeled, classes):
		self.labeled = labeled
		self.unlabeled = unlabeled
		self.classes = classes

		#initializing all values of the class dictionary to 0
		#the classes are the keys
		for key in self.classes:
			self.classDict[key] = 0

	def distance(self, point):
		distances = [] #holds distances of each point to the new point
		classSeries = self.labeled['class'] #a series of only the class for each point
		#to subtract a labeled point from an unlabeled, we cannot have classes
		labeled = self.labeled.drop('class', axis=1)

		#iterate through each row in the labeled (each row is a point), subtracting it
		#from the unlabeled point and getting the distance
		for i, series in labeled.iterrows():
			newSeries = point.subtract(series)
			#map applies a function to all elements in a series.
			#here we square each element, preparing to get the final distance
			newSeries = newSeries.map(lambda x: x**2)
			toRoot = newSeries.sum() #the number to root
			final = sqrt(toRoot) #the final distance from the labeled point to the unlabeled
			distances.append(final)
		#make a series from the list of distances with indices matching the classSeries
		distanceSeries = Series(distances)
		distanceSeries.index = classSeries.index

		#fill our classFrame with the class of each labeled point and its
		#distance from the unlabeled point
		self.classFrame = concat([classSeries, distanceSeries], axis=1)
		self.classFrame.columns = ['class', 'distances']

	def classifyOne(self, point, index):
		self.distance(point) #set up the classFrame for this method
		self.classFrame = self.classFrame.sort_values('distances')

		#take the first K classes of the sorted classFrame and count the
		#frequency of each class
		for index, row in self.classFrame.head(n=self.K).iterrows():
			thisClass = row['class'] #class of current row
			for type in self.classes:
				#add 1 to the frequency (in classDict) of whichever class this is
				if thisClass == type:
					self.classDict[thisClass] += 1
		#order the dict from highest to lowest frequency and return the highest
		answer = None
		for key in sorted(self.classDict, key=self.classDict.get, reverse=True):
			if answer == None:
				answer = key
			print(str(index) + " " + key + " {0:.0f}%".format((float(self.classDict[key])/float(self.K))*100), end=" ")
		print()
		return answer

	def classifyAll(self):
		answers = [] #list of classifications for each unlabeled point
		#go through each unlabeled point and classify it using classifyOne()
		for index, point in self.unlabeled.iterrows():
			#add result of classification to list of answers
			answers.append(self.classifyOne(point, index))
			#reset frequency count of each class for the next point's classification
			for key in self.classes:
				self.classDict[key] = 0

		#insert answers into the unlabeled data so we can return it
		answerSeries = Series(answers, index=self.unlabeled.index)
		self.unlabeled['class'] = answerSeries
		return self.unlabeled

	def accuracy(self, response, solution):
		tries = 0 #number of points we classified
		correct = 0 #number of points we classified correctly
		#go through our answers and increment "correct" each time we got
		#a point right
		print(response)
		for index, row in response.iterrows():
			tries += 1
			if row['class'] == solution['class'][index]:
				correct += 1
			else:
				print(str(index) + " Your answer: " + row['class'] + "   Correct answer: " + solution['class'][index])
		print(str(correct) + " out of " + str(tries) + " correct answers")
		print("{0:.0f}%".format((float(correct)/float(tries))*100) + " accuracy")
