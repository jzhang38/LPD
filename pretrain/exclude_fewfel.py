
import json
fewRel_val = 'finetune/data/val_wiki.json'
fewRel_train = 'finetune/data/train_wiki.json'
fewRel_test = 'finetune/data/test_wiki_input-5-1.json'
with open(fewRel_val) as few_Rel_val_file:
    few_Rel_val_dict: dict = json.load(few_Rel_val_file)
    few_Rel_val_dict = few_Rel_val_dict.keys()
with open(fewRel_train) as few_Rel_train_file:
    few_Rel_train_dict = json.load(few_Rel_train_file)
    few_Rel_train_dict = few_Rel_train_dict.keys()
    
few_Rel_test_dict = set()
with open(fewRel_test) as few_Rel_test_file:
    fewRel_test_js = json.load(few_Rel_test_file)
    for episode in fewRel_test_js:
        few_Rel_test_dict = set(episode["relation"]) | few_Rel_test_dict


few_rel_relation = few_Rel_val_dict | few_Rel_train_dict | few_Rel_test_dict
        
print("overlapping relation types: %d" % len(few_rel_relation))
to_save = {}
with open('data/CP/rel2scope.json') as file:
    pretrain_data = json.load(file)
    for k, v in pretrain_data.items():
        if k not in few_rel_relation:
            to_save[k] = v

with open('data/CP/rel2scope_excluded.json', 'w') as file:
    print("remaining relation types: %d" % len(to_save))
    json.dump(to_save, file)
    
