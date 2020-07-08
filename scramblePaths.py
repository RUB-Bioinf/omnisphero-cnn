import re


def filter_experiment_name(path, id):
    regex = '(\\W+\\w+)+_trainingData_\\w+\\/$'
    name = 'unknown' + str(id)
    m = re.match(regex, path)
    if m:
        m = m.group(1)
        m = m[1:]
        name = m
    return name


# Input: Array of paths that will be scrambled to be iterated over and train a CNN for each iteration.
# pathCandidateList: Input List validation_count: Based on the input list, how many of these entries shall be
# declared as validation data? If this entry is zero, only one iteration is made, with the first entry from the list
# taken as validation data. predict_count: Based on the input list, how many of these entries would you later like to
# predict on? These will be neither training, nor validation data and determine the 'label' parameter.

def scramble_paths(path_candidate_list: [str], validation_count: int, test_count: int):
    l = len(path_candidate_list)
    res = []

    if validation_count < 0:
        raise Exception('The requested amount of validation data is zero or less: ' + str(validation_count))

    if test_count < 0:
        raise Exception('The requested amount of prediction data is netgative: ' + str(test_count))

    if validation_count == 0:
        round = {}
        candidates = path_candidate_list.copy()
        candidates = candidates[::-1]

        val = []
        val.append(candidates.pop())

        test = []
        test.append(candidates.pop())

        round['label'] = 'custom'
        round['train'] = candidates
        round['val'] = val
        round['test'] = test

        res.append(round)
        return res

    for i in range(0, l, max(test_count, 1)):
        for j in range(test_count):
            path_candidate_list.append(path_candidate_list.pop(0))
        round = {}

        candidates = path_candidate_list.copy()
        train = []
        test = []
        rest = []
        val = []

        for j in range(l - (test_count + validation_count)):
            train.append(candidates.pop(0))

        for j in range(0, validation_count):
            val.append(candidates.pop(0))
        test = candidates[0]

        label = ''
        label_list = candidates
        if test_count == 0:
            label_list = val

        for j in range(len(label_list)):
            c = label_list[j]
            name = filter_experiment_name(c, j * i)
            # name = candidates[j]
            label = label + '_' + name
        label = label[1:]

        round['label'] = label
        round['train'] = train
        round['val'] = val
        round['test'] = test

        # print('Scramble round ' + str(i) + '. Validation: ' + filter_experiment_name(val[0], 0) + ' on label ' +
        # label)

        res.append(round)
    return res


def main():
    training_path_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    print("Running an example to scramble those elements: " + str(training_path_list))

    scramble_result = scramble_paths(path_candidate_list=training_path_list, validation_count=2, test_count=1)
    example_label = None
    example_train = None
    example_val = None
    example_test = None

    for n in range(len(scramble_result)):
        scrambles = scramble_result[n]
        example_label = scrambles['label']
        example_train = scrambles['train']
        example_val = scrambles['val']
        example_test = scrambles['test']

    print("Label: " + str(example_label))
    print("Train: " + str(example_train))
    print("Validation: " + str(example_val))
    print("Test: " + str(example_test))
    print('==============')


if __name__ == "__main__":
    main()
