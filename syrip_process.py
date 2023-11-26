import numpy as np
import json

with open('data/SyRIP_2d_gt/train200/person_keypoints_train_infant.json') as f:
    pose2d = json.load(f)


name_map = np.load('./data/survey_data/img_name700_map.npy')

train = {}
test = {}


real_test = []
for i in range(len(pose2d['images'])):
    real_test.append(pose2d['images'][i]['file_name'].split('/')[-1])
    
    
for idx, i in enumerate(name_map):
    if i[1] not in real_test:
        train[i[0]] =[i[1],idx]
    else:
        test[i[0]] = [i[1],idx]

        
np.save('test_rysip.npy',test)
np.save('train_rysip.npy',train)


with open('data/SyRIP_2d_gt/validate500/person_keypoints_validate_infant.json') as f:
    j1 = json.load(f)



dic = {}

for i in range(len(j1['images'])):
    dic[j1['images'][i]['file_name']] = {}
    dic[j1['images'][i]['file_name']]['h'] = j1['images'][i]['height']
    dic[j1['images'][i]['file_name']]['w'] = j1['images'][i]['width']
    dic[j1['images'][i]['file_name']]['bbox'] = j1['annotations'][i]['bbox']
    dic[j1['images'][i]['file_name']]['keypoints'] = np.array(j1['annotations'][i]['keypoints']).reshape((-1,3))
    
    
np.save('test_pose2d.npy',dic)


with open('data/SyRIP_2d_gt/validate500/person_keypoints_train_infant.json') as f:
    j1 = json.load(f)
    
dic = {}

for i in range(len(j1['images'])):
    dic[j1['images'][i]['file_name']] = {}
    dic[j1['images'][i]['file_name']]['h'] = j1['images'][i]['height']
    dic[j1['images'][i]['file_name']]['w'] = j1['images'][i]['width']
    dic[j1['images'][i]['file_name']]['bbox'] = j1['annotations'][i]['bbox']
    dic[j1['images'][i]['file_name']]['keypoints'] = np.array(j1['annotations'][i]['keypoints']).reshape((-1,3))

np.save('train_pose2d.npy',dic)
    
