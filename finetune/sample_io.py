import json
import sys
import random

if(len(sys.argv) != 7):
    print("Usage: python sample_io.py filename.json size N K seed input/output")
filename = sys.argv[1]
size = int(sys.argv[2])
N = int(sys.argv[3])
K = int(sys.argv[4])
seed = int(sys.argv[5])
io = sys.argv[6]
random.seed(seed)

whole_division = json.load(open(filename))
relations = whole_division.keys()

input_data = []
output_data = []
for i in range(size):
    sampled_relation = random.sample(relations, N)
    target = random.choice(range(len(sampled_relation)))
    output_data.append(target)
    target_relation = sampled_relation[target]
    meta_train = [random.sample(whole_division[i], K) for i in sampled_relation]
    meta_test = random.choice(whole_division[target_relation])
    input_data.append({"meta_train": meta_train, "meta_test": meta_test})

if(io == "input"):
    json.dump(input_data, sys.stdout)
else:    
    json.dump(output_data, sys.stdout)