import torch
import json
import models
import numpy as np
import sys
from torch.autograd import Variable
from fewshot_re_kit.data_loader import JSONFileDataLoader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder 
from models.proto import Proto

input_filename = sys.argv[1]

max_length = 40
test_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
framework = FewShotREFramework(None, None, test_data_loader)
sentence_encoder = CNNSentenceEncoder(test_data_loader.word_vec_mat, max_length)
model = Proto(sentence_encoder)
framework.set_model(model, 'checkpoint/proto.pth.tar')

content = json.load(open(input_filename))

ans = []
for episode in content:
	N = len(episode['meta_train'])
	K = len(episode['meta_train'][0])
	Q = 1
	support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
	query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
	for K_relation in episode['meta_train']:
		support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
		for single_instance in K_relation:
			cur_ref_data_word, data_pos1, data_pos2, data_mask, data_length = test_data_loader.lookup(single_instance)
			support_set['word'].append(cur_ref_data_word)
			support_set['pos1'].append(data_pos1)
			support_set['pos2'].append(data_pos2)
			support_set['mask'].append(data_mask)
		support['word'].append(support_set['word'])
		support['pos1'].append(support_set['pos1'])
		support['pos2'].append(support_set['pos2'])
		support['mask'].append(support_set['mask'])
	for K_relation in range(N):
		cur_ref_data_word, data_pos1, data_pos2, data_mask, data_length = test_data_loader.lookup(episode['meta_test'])
		query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
		query_set['word'].append(cur_ref_data_word)
		query_set['pos1'].append(data_pos1)
		query_set['pos2'].append(data_pos2)
		query_set['mask'].append(data_mask)
		query['word'].append(query_set['word'])
		query['pos1'].append(query_set['pos1'])
		query['pos2'].append(query_set['pos2'])
		query['mask'].append(query_set['mask'])

	support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, max_length))
	support['pos1'] = Variable(torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, max_length)) 
	support['pos2'] = Variable(torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, max_length)) 
	support['mask'] = Variable(torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, max_length)) 
	query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, max_length)) 
	query['pos1'] = Variable(torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, max_length)) 
	query['pos2'] = Variable(torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, max_length)) 
	query['mask'] = Variable(torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, max_length)) 

	logits, pred = framework.predict(support, query, N, K, Q)
	ans.append((int)(pred.numpy()[0]))

json.dump(ans, sys.stdout)