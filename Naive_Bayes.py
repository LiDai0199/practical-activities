import numpy as np
from scipy.stats import norm


def load_data(training_filename, testing_filename):

	# get testing data
	testing_file = open(testing_filename, "r")

	test_vector = []  # [ [x1], [x2], [x3], ... ] -> str
	for testing_line in testing_file:
		# get the numbers of the testing data: ["0.34", "0.25", ...]
		testing_num = testing_line.strip().split(',')

		test_vector.append(testing_num)

	# get the training data
	training_file = open(training_filename, "r")

	training_vector = []  # [ [x1], [x2], [x3], ... ] -> str
	for training_line in training_file:
		training_line_data = training_line.strip().split(',')
		training_vector.append(training_line_data)
	
	return training_vector, test_vector
	

def classify_nb(training_filename, testing_filename):
	
	result = []

	training_vector = load_data(training_filename, testing_filename)[0]
	test_vector = load_data(training_filename, testing_filename)[1]

	# seperate yes and no
	yes_ls = [[] for i in range(len(training_vector[0]) - 1)]  # [ [para1], [para2], [para3], ... ] -> float
	no_ls = [[] for i in range(len(training_vector[0]) - 1)]  # -1 because we do not need to record "yes"/"no"

	num_yes = 0
	num_no = 0

	for item in training_vector:
		if item[-1].lower() == "yes":
			num_yes += 1
			for i in range(len(item) - 1):
				yes_ls[i].append(float(item[i]))

		elif item[-1].lower() == "no":
			num_no += 1
			for i in range(len(item) - 1):
				no_ls[i].append(float(item[i]))

	yes_np = []  # [ [para1], [para2], [para3], ...] -> float
	no_np = []

	for ls in yes_ls:
		yes_np.append(np.array(ls))

	for ls in no_ls:
		no_np.append(np.array(ls))

	# calculate mean and sd
	mean_yes = []  # [mean1, mean2, mean3, ...] -> float
	mean_no = []

	sd_yes = []
	sd_no = []

	for vector in yes_np:
		mean_yes.append(np.mean(vector))
		sd_yes.append(np.std(vector))

	for vector in no_np:
		mean_no.append(np.mean(vector))
		sd_no.append(np.std(vector))

	# caculate probability
	p_yes = num_yes / (num_yes + num_no)
	p_no = num_no / (num_yes + num_no)

	for vector in test_vector:  # -> str
		p_predict_yes = 1 * p_yes
		p_predict_no = 1 * p_no

		for i in range(len(vector)):
			density_yes = norm.pdf(float(vector[i]), loc=mean_yes[i], scale=sd_yes[i])
			p_predict_yes *= density_yes

		for i in range(len(vector)):
			density_no = norm.pdf(float(vector[i]), loc=mean_no[i], scale=sd_no[i])
			p_predict_no *= density_no

		if p_predict_yes >= p_predict_no:
			result.append("yes")

		else:
			result.append("no")
	return result
