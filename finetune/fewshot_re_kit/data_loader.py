from matplotlib import use
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from tqdm import tqdm


class FewRelPredictionDataset(data.Dataset):
    def __init__(self, filepath, encoder,label_mask_prob=0.0):
       
        
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(filepath))
        self.pubmed = False
        if 'pubmed' in filepath:
            self.pubmed = True
   
        self.encoder = encoder
        property2description  = json.load(open('./data/property2description.json'))
        self.label2desc = {}
        for item in property2description:
            self.label2desc[item['id']] = [item['label'], item['description']]
        del property2description
        self.data = []
        self.label_mask_prob =label_mask_prob
        self.__prepare_data__()

    def __prepare_data__(self): 
        content = self.json_data   
        sup_count = 0
        query_count = 0 
        for episode in tqdm(content):
            N = len(episode['meta_train'])
            K = len(episode['meta_train'][0])
            Q = 1
            support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
            query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
            for K_relation, support_label in zip(episode['meta_train'], episode['relation']):
                for single_instance in K_relation:
                    show = self.__additem__(support, single_instance, support_label, True)
                    if sup_count  < 50:
                        print('Support')
                        sup_count+=1
                        print(show)
         
            show = self.__additem__(query, episode['meta_test'], None, False)
            if query_count < 10:
                print("Query:")
                query_count+=1
                print(show)
            self.data.append([support, query])
        
    def __additem__(self, d, item, classname, add_relation_desc):
        if add_relation_desc and random.random() >= self.label_mask_prob:
            if self.pubmed:
                label_des = classname.split('_')
            else:

                label_des = self.label2desc[classname][1] 
                label_des = label_des.split()
            token_ls = label_des + [':'] + item["tokens"]
            head_pos = []
            tail_pos = []
            for i in item['h'][2][0]:
                head_pos.append(i+len(label_des)+1)
            for i in item['t'][2][0]:
                tail_pos.append(i+len(label_des)+1)
        else:
            token_ls = item["tokens"]
            head_pos = item['h'][2][0]
            tail_pos = item['t'][2][0]

        word, pos1, pos2, mask = self.encoder.tokenize(token_ls,
            head_pos,
            tail_pos)
        
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

        return token_ls

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_fn_prediction(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
      
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
  
    return batch_support, batch_query
        
class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, args, name, encoder, N, K, Q, root, label_mask_prob=0.0):
        self.name = name
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))


        self.semeval = False
        self.pubmed = False
        if 'semeval' in path:
            self.semeval = True
        if 'pubmed' in path:
            self.pubmed = True
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.debug = args.debug



 

        self.label_mask_prob = label_mask_prob
        self.encoder = encoder
        property2description  = json.load(open('./data/property2description.json'))
        self.label2desc = {}
        for item in property2description:
            self.label2desc[item['id']] = [item['label'], item['description']]
        del property2description




    def __additem__(self, d, item, classname, add_relation_desc):

        if add_relation_desc and random.random() > self.label_mask_prob:
            if self.semeval:
               
                index = list(classname).index('(')
                label_des =list(classname)[:index]
                new = ""
                for x in label_des:
                    new += x 
                label_des = new
                label_des = label_des.split('-')
            elif self.pubmed:
                label_des = classname.split('_')
            else:

                label_des = self.label2desc[classname][1] 
                label_des = label_des.split()
  
            token_ls = label_des + [':'] + item["tokens"]
            head_pos = []
            tail_pos = []


            for i in item['h'][2][0]:
                head_pos.append(i+len(label_des)+1)
            for i in item['t'][2][0]:
                tail_pos.append(i+len(label_des)+1)
        else:
            token_ls = item["tokens"]
            head_pos = item['h'][2][0]
            tail_pos = item['t'][2][0]

        word, pos1, pos2, mask = self.encoder.tokenize(token_ls,
            head_pos,
            tail_pos)
        
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

        return token_ls


  

    def __getitem__(self, index):
        target_classes_id = random.sample(range(len(self.classes)), self.N)
        target_classes = [self.classes[class_id] for class_id in target_classes_id]
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        mlp_label = []
    
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))
     
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                
                if count < self.K:
                    show = self.__additem__(support_set, self.json_data[class_name][j], class_name, add_relation_desc=True)
                    if self.debug and self.label_mask_prob != 0:
                        print("Support")
                        print(show)
                else:
                    show = self.__additem__(query_set, self.json_data[class_name][j], class_name, add_relation_desc=False)
                    if self.debug and self.label_mask_prob != 0:
                        print("Query")
                        print(show)
                count += 1

            query_label += [i] * self.Q
            mlp_label += [target_classes_id[i]] * self.K


        return support_set, query_set, query_label, mlp_label
    
    def __len__(self):
        return 1000000000
    
    
    
    
    
    
    
    
    
    
    
    


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    batch_mlp_label = []
    support_sets, query_sets, query_labels, mlp_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
        batch_mlp_label += mlp_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    batch_mlp_label =  torch.tensor(batch_mlp_label)
    return batch_support, batch_query, batch_label, batch_mlp_label




def get_loader(args, name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn,  root='./data', label_mask_prob=1.0, ):

    dataset = FewRelDataset(args, name, encoder, N, K, Q,  root, label_mask_prob)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)






