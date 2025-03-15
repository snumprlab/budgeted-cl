import json
import jsonlines
from collections import defaultdict
import numpy as np
import os
import pickle

dataset = "Bongard-OpenWorld"
seed = 5
fixed_seed = 2024
num_tasks = 5
data_type = "generated"
split_record_file_name = f"{dataset}_split_record.pkl"
# split_record_file_name = f"{dataset}_split_record.pkl"

train_dict = defaultdict(list)
test_dict = defaultdict(list)
train_count = defaultdict(int)
test_count = defaultdict(int)

### Jsonl ###
with jsonlines.open(f"train_{data_type}.jsonl") as f:
    for line in f.iter():
        train_count[line["commonSense"]] += 1
        train_dict[line["commonSense"]].append(line)
        
with jsonlines.open("test.jsonl") as f:
    for line in f.iter():
        test_count[line["commonSense"]] += 1
        test_dict[line["commonSense"]].append(line)

most_num_key = '0'
train_second_most = sorted(train_count.values(), reverse=True)[1]
test_second_most = sorted(test_count.values(), reverse=True)[1]
train_count[most_num_key] = train_second_most
test_count[most_num_key] = test_second_most
train_dict[most_num_key] = np.random.choice(train_dict[most_num_key], size=train_second_most, replace=False)
test_dict[most_num_key] = np.random.choice(test_dict[most_num_key], size=test_second_most, replace=False)


print()
print("###########")
print("train count")
print(train_count)
print("test count")
print(test_count)
print("###########")
print()

categories = np.array(list(test_count.keys()))
### task split load/generate ### 
if os.path.isfile(os.path.join("ma", split_record_file_name)):
    with open(file=os.path.join("ma", split_record_file_name), mode='rb') as f:
        task_splits=pickle.load(f)[seed]["task_splits"]
else:
    np.random.seed(seed)
    np.random.shuffle(categories)
    task_splits = np.split(categories, num_tasks)
print(task_splits)

train_data_list = []
test_data_list = []
train_eval_point = []
test_num = []
for task_split in task_splits:
    np.random.seed(fixed_seed)
    train_task_data = np.concatenate([train_dict[task] for task in task_split])
    np.random.seed(fixed_seed)
    test_task_data = np.concatenate([test_dict[task] for task in task_split])
    np.random.shuffle(train_task_data)
    train_eval_point.append(len(train_task_data))
    test_num.append(len(test_task_data))
    train_data_list.extend(train_task_data)
    test_data_list.extend(test_task_data)
    
with open(f"{dataset}_train_seed{seed}.json", "w") as json_file:
    json.dump(train_data_list, json_file)


if data_type == "ma":
    with open(f"{dataset}_test.json", "w") as json_file:
        json.dump(test_data_list, json_file)

    ##### split record store #####
    if os.path.isfile(split_record_file_name):
        with open(file=split_record_file_name, mode='rb') as f:
            pickle_dict=pickle.load(f)
    else:
        pickle_dict = defaultdict(dict)

    pickle_dict[seed]["num_train_sample"] = train_count
    pickle_dict[seed]["num_test_sample"] = test_count
    pickle_dict[seed]["task_splits"] = task_splits
    pickle_dict[seed]["train_eval_point"] = train_eval_point
    print(pickle_dict)
    with open(file=split_record_file_name, mode='wb') as f:
        pickle.dump(pickle_dict, f)

    
