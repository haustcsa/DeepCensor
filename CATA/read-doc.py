import requests
import adv_cot as ac
import mapping as map
import time
import json
#用于读取构造的advcot
dir_path=ac.file_name+'.txt'
attack_type =ac.attack_type
sting=0
for line in open(dir_path):
    sting=eval(line)



headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# sst
# # "positive": 1,
# # "negative": 0
data_map=map.Mapping(dataset=ac.dataset)

resa_output=[]
label=[]
sucadv=0
res_output=[]
temp=0

if attack_type != "adv-glue":
    for i in sting:
        #接续异常
        if temp==0:
            # print(i[0],i[1])
            output_adv = query({
                "inputs": "Q:"+i[1],
                "wait_for_model": True
            })
            output_ori=query({
                "inputs": "Q:"+i[0],
                "wait_for_model": True
            })
            label_ori=data_map.dataset_mapping(i[2])
            # 判断是否存在正常访问，否则输出报错，
            try:
                # 可能会产生异常的代码块
                    a=output_adv[0]
                    b=output_ori[0]
            except :
                # 处理ZeroDivisionError异常的代码块
                print(output_adv, "\n", output_ori)



            fm="res-"+ac.file_name
            with open(fm+'.txt', 'a') as f:
                    # f.write("[")
                    #对抗，正常，原始标签
                    adv=output_adv[0]['generated_text'].split()[-1].replace(".","").lower()
                    f.write(adv)
                    f.write(",")
                    ori=output_ori[0]['generated_text'].split()[-1].replace(".","").lower()
                    f.write(ori)
                    f.write(",")
                    f.write(label_ori)
                    # f.write("]")
            with open(fm+'.txt', 'a') as f:
                    f.write('\n')
            resa_output.append(adv)
            res_output.append(ori)
            label.append(label_ori)
            print("adv_label:",adv,"\n","output_ori:",ori,"\n","origin_label:",data_map.dataset_mapping(i[2]))
        else:
            temp = temp + 1
else:
    for i in sting:
        # 接续异常
        if temp == 0:
            # print(i[0],i[1])
            output_adv = query({
                "inputs": "Q:" + i[1],
                "wait_for_model": True
            })
            label_ori = data_map.dataset_mapping(i[2])
            # 判断是否存在正常访问，否则输出报错，
            try:
                # 可能会产生异常的代码块
                a = output_adv[0]
                b = " "
            except:
                # 处理ZeroDivisionError异常的代码块
                print(output_adv, "\n")

            fm = "res-" + ac.file_name
            with open(fm + '.txt', 'a') as f:
                # f.write("[")
                # 对抗，正常，原始标签
                # 在UL  fs任务中会得到一个长句子的答案，即并未按照指令规定的范围输出
                adv = output_adv[0]['generated_text'].split()[-1].replace(".", "").lower()
                f.write(adv)
                f.write(",")
                ori = " "
                f.write(ori)
                f.write(",")
                f.write(label_ori)
                # f.write("]")
            with open(fm + '.txt', 'a') as f:
                f.write('\n')
            resa_output.append(adv)
            res_output.append(ori)
            label.append(label_ori)
            print("adv_label:", adv, "\n", "output_ori:", ori, "\n", "origin_label:", data_map.dataset_mapping(i[2]))
        else:
            temp = temp + 1

acca_count = sum([1 for x, y in zip(resa_output, label) if x == y])
acc_count = sum([1 for x, y in zip(res_output, label) if x == y])
asr_count=sum([1 for x, y in zip(resa_output, res_output) if x != y])
accarate=acca_count/len(label)
accrate=acc_count/len(label)
asr_=asr_count/len(label)
pdr=(acc_count-acca_count)/acc_count
print("对抗acc:",accarate,acca_count)
print("初始acc:",accrate,acc_count)
print("下降acc:",accrate-accarate)
print("ASR:",asr_,asr_count)
print("PDR:",pdr)

