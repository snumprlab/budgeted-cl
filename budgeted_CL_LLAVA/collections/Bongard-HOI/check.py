from collections import defaultdict
import json

with open('ma_splits/Bongard-HOI_train_seed1.json', 'r') as data:
    train_lists = json.load(data)

# with open('ma/5_set/Bongard-HOI_train_seed1.json', 'r') as data:
#     train_lists = json.load(data)

print("train_lists", len(train_lists))
seen_action_object = defaultdict(int)
for train_data in train_lists:
    for action_class, object_class in zip(train_data["action_class"], train_data["object_class"]):
        key = f'{train_data["action_class"][0]} {train_data["object_class"][0]}'.replace('_', ' ')
        seen_action_object[key] += 1
breakpoint()
print(seen_action_object)