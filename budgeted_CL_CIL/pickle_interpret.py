import pickle

with open('baseline_cls_std.pickle', 'rb') as handle:
    cls_std_list = pickle.load(handle)

with open('baseline_sample_std.pickle', 'rb') as handle:
    sample_std_list = pickle.load(handle)

print("cls_std_list")
print(cls_std_list)
print("sample_std_list")
print(sample_std_list)