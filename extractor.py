import re

with open('exps/_outputs/2023-02-05-10:23:29.txt') as f:

    lines = f.readlines()
    start = False

    for l in lines:
        
        if start and l.startswith('INFO - main - mar_val batches classDice:'):
            all_nums = re.findall("\d+\.\d+", l)
            # assert len(all_nums) == 1
            if len(all_nums) != 1:
                print(0)
                # print(l)
            else: print(float(all_nums[0]))

        if l.startswith('Evaluating organ 1(SPLEEN)'):
            start=False
        elif l.startswith('Evaluating organ 2(KID_R)'):
            start=False
        elif l.startswith('Evaluating organ 3(KID_l)'):
            start=False
        elif l.startswith('Evaluating organ 6(LIVER)'):
            start=True
        