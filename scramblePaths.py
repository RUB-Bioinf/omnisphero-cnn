import re

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

def scramblePaths(pathCandidateList,dropRate):
    l = len(pathCandidateList)

    res = []
    for i in range(l):
    	pathCandidateList.append(pathCandidateList.pop(0))
    	round = {}

    	candidates = pathCandidateList.copy()
    	train = []
    	rest = []
    	val = []

    	for j in range(l-(dropRate+1)):
    		train.append(candidates.pop(0))

    	val.append(candidates.pop(0))

    	label = ''
    	for j in range(len(candidates)):
    		c = candidates[j]
    		name = filterExperimentName(c,j*i)
    		label = label + '_' + name
    	label = label[1:]

    	round['label'] = label
    	round['train'] = train
    	round['val'] = val

    	print('Scramble round '+str(i)+'. Validation: '+filterExperimentName(val[0],0)+' on label '+label)

    	res.append(round)
    return res

#scrableResult = scramblePaths(training_path_list,2)
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