import json
import random
import pickle
import numpy as np
from collections import defaultdict
import random
import os.path

dataset = "Bongard-HOI"
types = "generated"
seed = 5
num_tasks = 5
split_record_file_name = f"{dataset}_{types}_split_record.pkl"

with open(f"train_{types}.json", "r") as data:
    train_lists = json.load(data)
    
with open("test.json", "r") as data:
    test_lists = json.load(data)

print("total test length", len(test_lists))
print("total train length", len(test_lists))

train_action_lists = []
train_object_lists = []
train_action_object_pair_lists = []
train_action_object_pair_dict = defaultdict(list)
train_count_dict = defaultdict(int)

test_action_lists = []
test_object_lists = []
test_action_object_pair_lists = []
test_action_object_pair_dict = defaultdict(list)
test_count_dict = defaultdict(int)

for train_data in train_lists:
    train_action_lists.append(train_data["action_class"][0])
    train_object_lists.append(train_data["object_class"][0])
    train_action_object_pair_lists.append((train_data["action_class"][0], train_data["object_class"][0]))
    key = f"{train_data['action_class'][0]}_{train_data['object_class'][0]}"
    train_data["action_object"] = key
    train_data["action"] = train_data['action_class'][0]
    train_count_dict[key] += 1
    train_action_object_pair_dict[key].append(train_data)

for test_data in test_lists:
    test_action_lists.append(test_data["action_class"][0])
    test_object_lists.append(test_data["object_class"][0])
    test_action_object_pair_lists.append((test_data["action_class"][0], test_data["object_class"][0]))
    key = f"{test_data['action_class'][0]}_{test_data['object_class'][0]}"
    test_count_dict[key] += 1
    test_action_object_pair_dict[key].append(test_data)
    
train_action_lists = list(set(train_action_lists))
train_object_lists = list(set(train_object_lists))
train_action_object_pair_lists = list(set(train_action_object_pair_lists))

test_action_lists = list(set(test_action_lists))
test_object_lists = list(set(test_object_lists))
test_action_object_pair_lists = list(set(test_action_object_pair_lists))

print()
print("##### Statistics #####")
print("train_action_lists", len(train_action_lists))
print("train_object_lists", len(train_object_lists))
print("train_action_object_pair_lists", len(train_action_object_pair_lists))
print("train_count_dict")
print(train_count_dict)
print("------------------------------")
print("test_action_lists", len(test_action_lists))
print("test_object_lists", len(test_object_lists))
print("test_action_object_pair_lists", len(test_action_object_pair_lists))
# print("test_count_dict")
# print(test_count_dict)
print()

intersection_pairs = np.array(list(set(list(train_count_dict.keys())).intersection(set(list(test_count_dict.keys()))) - set(["jump_motorcycle", "ride_motorcycle", "eat_at_dining_table", "hold_dog", "sit_on_bed", "ride_bicycle"])))
intersection_pairs.sort()
print("# of train-test intersection", len(intersection_pairs))

### task split load/generate ### 
if os.path.isfile(os.path.join("ma", split_record_file_name)):
    with open(file=os.path.join("ma", split_record_file_name), mode='rb') as f:
        task_splits=pickle.load(f)[seed]["task_splits"]
else:
    np.random.seed(seed)
    np.random.shuffle(intersection_pairs)
    task_splits = np.split(intersection_pairs, num_tasks)

train_data_list = []
test_data_list = []
train_eval_point = []
test_num = []
for task_split in task_splits:
    train_task_data = np.concatenate([train_action_object_pair_dict[task] for task in task_split])
    test_task_data = np.concatenate([test_action_object_pair_dict[task] for task in task_split])
    np.random.shuffle(train_task_data)
    train_eval_point.append(len(train_task_data))
    test_num.append(len(test_task_data))
    train_data_list.extend(train_task_data)
    test_data_list.extend(test_task_data)
    
with open(f"{dataset}_train_seed{seed}.json", "w") as json_file:
    json.dump(train_data_list, json_file)
    
with open(f"{dataset}_test.json", "w") as json_file:
    json.dump(test_data_list, json_file)

##### split record store #####
if os.path.isfile(split_record_file_name):
    with open(file=split_record_file_name, mode='rb') as f:
        pickle_dict=pickle.load(f)
else:
    pickle_dict = defaultdict(dict)

pickle_dict[seed]["task_splits"] = task_splits
pickle_dict[seed]["train_eval_point"] = train_eval_point
print(pickle_dict)
with open(file=split_record_file_name, mode='wb') as f:
    pickle.dump(pickle_dict, f)


