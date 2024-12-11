import pandas as pd
from model.GlandModel import GlandGrading
import pickle
import torch
from model.Models import NestedUNet

model = NestedUNet()
model.load_state_dict(torch.load("./unet.pth"))

annotation = pd.read_csv('ds/images/Grade.csv', delimiter=',')

res = []
for line in annotation.iterrows():
    l = line[1]
    name = l['name']
    grade = l[' grade (GlaS)']
    label = 0
    if grade == ' malignant':
        label = 1
    instance = GlandGrading(name, label, model)
    instance.extract_roi()
    instance.get_descriptors()
    res.append(instance)

with open('ds/data.pkl', 'wb') as f:
    pickle.dump(res, f)



