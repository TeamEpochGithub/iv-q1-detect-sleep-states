# read local requirements.txt
# also read kaggle requirments
# return a list of all packages (exluding the versions) that are on the local requirments but are not on the kaggle requirments


local_reqs = []
with open('local_requirements.txt', 'r', encoding='utf-16le') as f:
    for i, line in enumerate(f):
        if '=' in line:
            local_reqs.append(line.split('=')[0].strip())
        else:
            local_reqs.append(line.strip())

kaggle_reqs = []
with open('kaggle_requirements.txt', 'r', encoding='utf-16le') as f:
    for line in f:
        if '=' in line:
            kaggle_reqs.append(line.split('=')[0].strip())
        else:
            kaggle_reqs.append(line.strip())

# make both sets
local_reqs = set(local_reqs)
kaggle_reqs = set(kaggle_reqs)

# find the difference
diff = local_reqs - kaggle_reqs
# print(diff)

# now take the pytorch segmentation models requirments
smp_reqs = 'torchvision>=0.5.0 pretrainedmodels==0.7.4 efficientnet-pytorch==0.7.1 timm==0.9.7 tqdm pillow six'

for req in smp_reqs.split(' '):
    if req.split('=')[0] not in diff:
        print(req.split('=')[0])
