# 用于实现adv-cot的构造
import cot_generate as cg
import json

vic_model="BLOOM"#/GPT/GPT-J
dataset = "qqp"#mnli qqp qnli rte sst2
lang = "en"
flag = 1# FsCot":0;ZsCOT:1
attack_name ="checklist"#
attack_type = "sentence-attack"#
sub_model = "bert"#
sub_model="adv"


file_name = ''
# 生成文件
temp = []

ZsCot = ''
FsCOT = ''

# 读取文件

dir_path = attack_type + "\\" + dataset + "-" + sub_model + "-" + attack_name + ".txt"
# dir_path='char-attack\\sst2-bert-textbugger.txt'

# sting = 0
for line in open(dir_path):
    sting = eval(line)

# 获取文件名称
clean_path = dir_path.replace('\\', '-')
directory = clean_path.split('.')[0]
for line in open(dir_path):
    sting = eval(line)
if flag:
    ZsCot = cg.ZScot(dataset=dataset, lang=lang, vic_model=vic_model)
    file_name = "cot-" + directory + "-" + vic_model + "ZS"
    if attack_type != "adv-glue":
        for i in sting:
            # 调用类中的方法
            datacot = ZsCot.ZS_cot(i[0])
            advdatacot = ZsCot.ZS_cot(i[1])
            temp.append([datacot, advdatacot, i[2]])
    else:
        for i in sting:
            # 调用类中的方法
            datacot = " "
            advdatacot = ZsCot.ZS_cot(i[1])
            temp.append([datacot, advdatacot, i[2]])
else:
    FsCOT = cg.FScot(dataset=dataset, lang=lang, vic_model=vic_model)
    file_name = "cot-" + directory + "-" + vic_model + "FS"
    if attack_type != "adv-glue":
        for i in sting:
            # 调用类中的方法
            datacot = FsCOT.FS_cot(i[0])
            advdatacot = FsCOT.FS_cot(i[1])
            temp.append([datacot, advdatacot, i[2]])
    else:
        for i in sting:
            # 调用类中的方法
            datacot = " "
            advdatacot = FsCOT.FS_cot(i[1])
            temp.append([datacot, advdatacot, i[2]])

with open(file_name + '.txt', 'w') as f:
    f.write(json.dumps(temp))
