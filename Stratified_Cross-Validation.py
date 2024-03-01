import random

def make_fold_data(file_to_read, file_to_write):
    # Load the data
    f = open(file_to_read, 'r')
    data = []
    for line in f:
        data.append(line.strip().split(','))

    # Split the data into yes and no examples
    yes_data = []
    no_data = []

    for row in data:
        # Check if the last column is 'yes'
        if row[-1] == 'yes':
            yes_data.append(row)
        # Check if the last column is 'no'
        elif row[-1] == 'no':
            no_data.append(row)


    # Shuffle the data
    random.shuffle(yes_data)
    random.shuffle(no_data)

    # Calculate the number of folds and the number of examples per fold
    folds = 10
    yes_per_fold = [len(yes_data) // folds] * folds
    no_per_fold = [len(no_data) // folds] * folds

    # Distribute the remaining examples
    for i in range(len(yes_data) % folds):
        yes_per_fold[i] += 1
    for i in range(len(no_data) % folds):
        no_per_fold[i] += 1

    # Open the output file
    f = open(file_to_write, 'w')
    # Iterate over the folds
    yes_start = 0
    no_start = 0
    for i in range(folds):

        # Write the fold name
        f.write('fold' + str(i+1) +'\n')

        # Write the yes examples for this fold
        yes_end = yes_start + yes_per_fold[i]
        for row in yes_data[yes_start:yes_end]:
            f.write(','.join(row) + '\n')

        # Write the no examples for this fold
        no_end = no_start + no_per_fold[i]
        for row in no_data[no_start:no_end]:
            f.write(','.join(row) + '\n')

        f.write('\n')

        # Update the start indices for the next fold
        yes_start = yes_end
        no_start = no_end

if __name__ == '__main__':
    make_fold_data("pima.csv", "pima-folds.csv")
