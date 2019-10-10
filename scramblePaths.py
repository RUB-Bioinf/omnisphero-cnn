import re
from sklearn.metrics import roc_curve, roc_auc_score, auc

# # === TEST DATA ==
# training_path_list = [
#         '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/',
# 		'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_neuron/'
#         #TODO
#             ]
# 
# validation_path_list = [
#         '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/'
#        ]
# 
# outPath = '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/results/';
# label = 'ayy';
# #=====

def filterExperimentName(path,id):
	regex = '(\\W+\\w+)+_trainingData_\\w+\\/$'
	name = 'unknown' + str(id)
	m = re.match(regex,path)
	if m:
		m = m.group(1)
		m = m[1:]
		name = m
	return name

# Input: Array of paths that will be scrambled to be iterated over and train a CNN for each iteration.
#	pathCandidateList: Input List
#   validation_count: Based on the input list, how many of these entries shall be declared as validation data?
#						If this entry is zero, only one iteration is made, with the first entry from the list taken as validation data.
#   predict_count: Based on the input list, how many of these entries would you later like to predict on? These will be neither training, nor validation data and determine the 'label' parameter.
def scramblePaths(pathCandidateList,validation_count,predict_count):
    l = len(pathCandidateList)
    res = []

    if validation_count < 0:
    	raise Exception('The requested amount of validation data is zero or less: ' + str(validation_count))

    if predict_count < 0:
    	raise Exception('The requested amount of prediction data is netgative: ' + str(predict_count))

    if validation_count == 0:
    	round = {}
    	candidates = pathCandidateList.copy()
    	candidates = candidates[::-1];

    	val = []
    	val.append(candidates.pop())

    	round['label'] = 'custom'
    	round['train'] = candidates
    	round['val'] = val

    	res.append(round)
    	return res

    #if predict_count == 0:
    #	round = {}
    #	train = pathCandidateList
    #	rest = []
    #	val = []
    #	for j in range(0,validation_count):
    #		val.append(train.pop(0))
	#
    #	round['label'] = 'All'
    #	round['train'] = train
    #	round['val'] = val
    #	res.append(round)
	#
    #	return res

    for i in range(0,l,max(predict_count,1)):
    	for j in range(predict_count):
    		pathCandidateList.append(pathCandidateList.pop(0))
    	round = {}

    	candidates = pathCandidateList.copy()
    	train = []
    	rest = []
    	val = []

    	for j in range(l-(predict_count+validation_count)):
    		train.append(candidates.pop(0))

    	for j in range(0,validation_count):
    		val.append(candidates.pop(0))

    	label = ''
    	labelList = candidates;
    	if predict_count == 0:
    		labelList = val

    	for j in range(len(labelList)):
    		c = labelList[j]
    		name = filterExperimentName(c,j*i)
    		#name = candidates[j]
    		label = label + '_' + name
    	label = label[1:]

    	round['label'] = label
    	round['train'] = train
    	round['val'] = val

    	print('Scramble round '+str(i)+'. Validation: '+filterExperimentName(val[0],0)+' on label '+label)

    	res.append(round)
    return res

#training_path_list = ['a','b','c','d','e','f','g']
#
#scrableResult = scramblePaths(pathCandidateList=training_path_list,validation_count=2,predict_count=1)
#for n in range(len(scrableResult)):
#	scrambles = scrableResult[n]
#	label = scrambles['label']
#	train = scrambles['train']
#	val = scrambles['val']
#
#	print('L:')
#	print(label)
#	print('T')
#	print(train)
#	print('V')
#	print(val)
#	print('==============')