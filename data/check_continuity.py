import json

cls_map = None
with open('/HDD/medical-data/ABD-CT/ABD-CT-Preprocessed/classmap_1.json', 'r') as fopen:
    cls_map =  json.load(fopen)
    fopen.close()


for clss, val in cls_map.items():
    print('Class:', clss)
    
    for vol, slices in val.items():
        print(vol)
        
        if len(slices) == 0:
            continue

        start = slices[0]
        for slc in slices:
            assert start == slc
            start += 1
        
        print('Done')
    
    print('-------------')
    
