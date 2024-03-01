import numpy as np


def classify_nn(training_filename, testing_filename, k):
	result = []

	# read file

	testing_file = open(testing_filename, "r")
	testing_points = []

	for testing_line in testing_file:

		# get the numbers of the testing data: ["0.34", "0.25", ...]
		testing_num = testing_line.strip().split(',')

		for i in range(len(testing_num)):  # change to float number type
			testing_num[i] = float(testing_num[i])

		# create an testing vector
		testing_points.append(np.array(testing_num))


	training_categories = []
	training_points = []
	training_file = open(training_filename, "r")  # Each time needs to read the training file again
	for training_line in training_file:

		training_line_data = training_line.strip().split(',')

		# copy the numerical part: ["0.34", "0.25", ...]
		training_num = training_line_data[:len(training_line_data) - 1]
		for i in range(len(training_num)):  # change to float number type
			training_num[i] = float(training_num[i])

		# copy the category: "yes" / "no"
		training_categories.append(training_line_data[-1].lower())

		# calculate distance
		training_points.append(np.array(training_num))

	distances = []
	for testing_point in testing_points:
		# get the training data and calculate the distance
		distance = []  # [ [distance1, classification1], [distance2, classification2] ]
		for training_point, training_category in zip(training_points, training_categories):
			sum_sq = np.sum(np.square(testing_point - training_point))
			dist = np.sqrt(sum_sq)

			distance.append([dist, training_category])
		distances.append(distance)

	for distance in distances:

		# figure out the classification
		distance_sorted = sorted(distance,
								 key=lambda x: x[0])  # [ [distance1, classification1], [distance2, classification2] ]

		num_neighbor = k
		num_yes = 0
		num_no = 0

		if num_neighbor <= len(distance_sorted):
			for i in range(num_neighbor):
				if distance_sorted[i][1] == "yes":
					num_yes += 1

				elif distance_sorted[i][1] == "no":
					num_no += 1

			if num_yes >= num_no:
				result.append("yes")
			else:
				result.append("no")

	return result
