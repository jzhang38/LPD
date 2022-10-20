import json

## 64
with open('finetune/fewshotRE/FewRel/data/train_wiki.json') as f:
    js = json.load(f)
    print(len(js))
    
## 698
with open('data/CP/rel2scope.json') as f:
    js = json.load(f)
    print(len(js))