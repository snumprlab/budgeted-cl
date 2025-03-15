import json

with open('collections/DomainNet/DomainNet_sdxl_diversified_sigma0_repeat1_init100_seed5.json', 'r') as f:
    data = json.load(f)

print(data['cls_dict'])
# class_list = []

# for sample in data['stream']:
#     class_name = sample['klass']
#     if class_name not in class_list:
#         print(f"Class {class_name}")
#         class_list.append(class_name)
    