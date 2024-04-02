import os
import sys,csv
from time import gmtime, strftime
from tqdm import tqdm
from core.db_wd import DBWikidata
import utils
import re
import operator
import pandas as pd
import collections
import gzip
import pickle
from numpy import nan 
from collections import defaultdict
db = DBWikidata()

def get_type():
    
    l_all_type = []
    content_into_list = []
    my_file = open(txtfile, "r")
    content = my_file.read()
    content_into_list = content.split("\n")
    for i in tqdm(content_into_list):
        l_type = db.get_instance_of(i)
        if l_type is not None:
            l_all_type = l_all_type + l_type
        else:
            continue
    all_type = sorted(set(l_all_type))
    with open(outfile, 'w') as f:
        for item in all_type:
        #for item in result:
            (f.write("%s\n" % item))

def txt_handle(tfile):
    my_file = open(tfile, "r")
    content = my_file.read()
    type_all = content.split("\n")
    #print(type_all)
    return type_all 

def get_type1(txtfile,outfile):
    list_type = []
    list_ = []
    l = txt_handle(txtfile)
    for i in tqdm(l):
        # j +=1
        list_i = re.split(r'["-->"]+', i)
        q1 = db.get_label(list_i[0])
        q2 = db.get_label(list_i[1])
        if 1 is not None:
            list_.append(1)
        else:
            list_.append(list_i[0])
        if 2 is not None:
            list_.append(2)
        else:
            list_.append(list_i[1])
        type_i ="-->".join(list_)
        list_type.append(type_i)
        list_ = []
        type_i = ""
    #print(list_type)
    with open(outfile, 'w') as f:
        for item in list_type:
            (f.write("%s\n" % item))

def get_max():
    dict = utils.read_json("/home/nhuvn/vir_wd/count_type/output/chart1_couple.json")
    l_val = dict.values()
    max_val = max(l_val)
    new_ma_val = max(dict.items(), key=operator.itemgetter(1))[0]
    print((new_ma_val))
    print(max_val)

def get_max_type(dict_type):
    a = utils.read_json(dict_type)
    # print(len(a))
    j = 0
    list_ = []
    dict_max = {}
    for i in tqdm(a.values()):
        j +=1
        if i.values():
            # print(len(i))
            m = max(i.values())
            #print(m)
            couple_type = get_key(m,i)
            print(f"{couple_type},{m}" )
        #     list_i = re.split(r'["-->"]+', couple_type)
        #     # if list_i[0] =="13406463":
        #     #     1 ="Wikimedia list article"
        #     # else:
        #     1 = db.get_label(list_i[0])
        #     # if list_i[1] =="13406463":
        #     #     1 ="Wikimedia list article"
        #     # else:
        #     2 = db.get_label(list_i[1])
        #     if 1 is not None:
        #         list_.append(1)
        #     else:
        #         list_.append(list_i[0])
        #     if 2 is not None:
        #         list_.append(2)
        #     else:
        #         list_.append(list_i[1])
        #     couple_type_label ="-->".join(list_)
        #     dict_max[couple_type_label] = m
        #     list_ = []
        #     couple_type_label =""
        #     m = 0

        # else:
        #     key = "1"
        #     dict_max[key] = 0
        #     continue
        # if j ==5:
        #     break
    # utils.write_dict_csv(dict_max,outfile)
   # print(len(dict_max))
    # print(dict_max)


def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
    

def get_len_max(dict_type):
    a = utils.read_json(dict_type)
    # print(len(a))
    j = 0
    list_ = []
    # dict_max = {}
    for key,val in tqdm(a.items()):
        j +=1
        label =db.get_label(key)
        if val.values():
            l = len(val)
            m = max(val.values())
            #print(m)
            couple_type = get_key(m,val)
            list_i = re.split(r'["-->"]+', couple_type)
            q1 = db.get_label(list_i[0])
            q2 = db.get_label(list_i[1])
            if 1 is not None:
                list_.append(1)
            else:
                list_.append(list_i[0])
            if 2 is not None:
                list_.append(2)
            else:
                list_.append(list_i[1])
            couple_type_label ="-->".join(list_)
            print(f"{label},{l},{couple_type_label},{m}" )
            list_ = []
        else:
            print(label)
            # print(0)
            continue

def get_type50(dict_type):
    a = utils.read_json(dict_type)
    # print(len(a))
    j = 0
    list_type50 = []
    dict_50 = {}
    list50 =[]
    for key,val in tqdm(a.items()):
        j +=1
        label =db.get_label(key)
        # print(len(val))
        # print(val)
        for k,v in val.items():
            if v >50:
                # print(k)
                # list_type50.append(k) 
                list_i = re.split(r'["-->"]+', k)
                q1 = db.get_label(list_i[0])
                q2 = db.get_label(list_i[1])
                if 1 is not None:
                    list_type50.append(1)
                else:
                    list_type50.append(list_i[0])
                if 2 is not None:
                    list_type50.append(2)
                else:
                    list_type50.append(list_i[1])
                # couple_type_label ="-->".join(list_)
                type_50 = "-->".join(list_type50)
                list50.append(type_50)
                list_type50 = []
                type_50 =""
        all_50 = ",".join(list50)
        dict_50[label] = all_50
    utils.write_dict_csv(dict_50,"/home/nhuvn/wikidb/ouput/type501.csv")

def commonlist():
    l1 = txt_handle("/home/nhuvn/count_type/output/type_relation_common_count_label.txt")
    l2 = txt_handle("/home/nhuvn/count_type/output/type_relation_common_label.txt")
    l3 = set(l1).intersection(set(l2))
    for i in l3:
        print(i)

def write_excel(dict1,outfile):
    df = pd.DataFrame(data=dict1, index=[0])
    df = (df.T)
    df.to_excel(outfile)

def type_left():
    # dict_type = utils.read_json("/home/nhuvn/count_type/output/probs50_evj.json")
    l = txt_handle("/home/nhuvn/vir_wd/code/tesst.txt")
    for key in l:
        print(f"{key},{db.get_label(key)}")

def type_label():
    csvfile = "/home/nhuvn/vir_wd/mapping_en.csv"
    data = pd.read_csv(csvfile)
    list_ents = data['wikidata_id'].tolist()
    # print(len(set(list_ents))) #6,4M
    set_ent = set(list_ents)
    with gzip.open('/home/nhuvn/dataset/6M_en.pkl.gz', 'wb') as f:
        pickle.dump(set_ent, f)


def read_pkl():
    list_type = utils.txt_handle("/home/nhuvn/dataset/600types.txt")
    with gzip.open('/home/nhuvn/dataset/all_id_ja.pkl.gz', 'rb') as f:
        data = pickle.load(f)
    # j = 0
    # print("_id,type")
    # for i in data:
    #     l = db.get_instance_of(i)
    #     # j +=1
    #     if l:
    #         print(f"{i},{l[0]}")
        # if j == 1000000:
        #     break
    dict_id = defaultdict(list)
    # list_id = []
    # # print(len(list_type))
    for j in tqdm(list_type):
        # k +=1
        for i in data:
            # print(i)
            a = db.get_instance_of(i)
            if a and (j in a):
                # print(a[0])
                if j in dict_id.keys():
                    dict_id[j].append(i)
                else:
                    dict_id[j] = [i]
                if len(a) ==1:
                    data.remove(i)
            else:
                continue
            
    #     dict_id [j] = list_id
    #     list_id = []
    #     # if k ==5:
        #     break
           
          
    # for i in tdm(list_type):
    #     for j in data:
    #         l = db.get_instance_of(j)
    #         if l and (i in l):
    #             list_id.append(j)
    #             data.remove(j)
    #         else:
    #             continue

    # list_ent = list(data_all)
    # list_ent_all = [item for item in list_ent if not(pd.isnull(item)==True)]
    # with gzip.open('/home/nhuvn/dataset/all_human_en.pkl.gz', 'rb') as f:
    #     data_human = pickle.load(f)
    # data = set(list_ent_all).difference(set(data_human))
    # print(len(data))
    # print(type(data))
    # # print(data.shape)
    # list_human = []
    # j = 0
    
    # # print(len(list_ent_final)) #6438267
    with gzip.open('/home/nhuvn/dataset/dict_600type_ja.pkl.gz', 'wb') as f:
        pickle.dump(dict_id, f)


def compare2():
    # dict_type_en = utils.read_json("/home/nhuvn/count_type/output/common_righttype_en.json")
    dict_type_ja = utils.read_json("/home/nhuvn/count_type/output/common_righttype_ja.json")
    # dict_type_vi = utils.read_json("/home/nhuvn/count_type/output/common_righttype_vi.json")
    dict_result = {}
    # l = []
    # for k,v in tdm(dict_type_en.items()):
    #     l.append(len(v))
    # print(sorted(l))
    # i = 0
    for key, value in tdm(dict_type_ja.items()):
        # i +=1
        new_value = {k:v for k,v in value.items() if v >0.5}
        dict_result[key] = new_value
    # dict_type = utils.read_json("/home/nhuvn/count_type/output/probs_of_type_vi.json")
    # dict_type = utils.read_json("/home/nhuvn/count_type/output/prob_right2left_50_en.json")
    l = txt_handle("/home/nhuvn/count_type/output/probs_of_type_right_evj.txt")
    dict_result1 = {}
    # i =0
    for type_right in tdm(l):
        for key, value in dict_result.items():
        # i +=1
        # new_value = {k:v for k,v in value.items() if v >0.5}
        # dict_result[key] = new_value
            if type_right == key:
                # new_value = {k:v for k,v in value.items() if v >0.5}
                dict_result1[type_right] = value
            else:
                continue
    # print(i)
    # print(len(dict_result))
    new_dict = {k:v for k,v in dict_result1.items()if v}
    # print(len(new_dict))
    for key,value in new_dict.items():
        for k,v in value.items():
            print(f"{key},{db.get_label(k),v}")
   
def compare1():
    dict_type_en = utils.read_json("/home/nhuvn/count_type/output/common_righttype_en.json")
    # dict_type_ja = utils.read_json("/home/nhuvn/count_type/output/common_righttype_ja.json")
    # dict_type_vi = utils.read_json("/home/nhuvn/count_type/output/common_righttype_vi.json")
    dict_result = {}
    # l = []
    # for k,v in tdm(dict_type_en.items()):
    #     l.append(len(v))
    # print(sorted(l))
    i = 0
    for key, value in tqdm(dict_type_en.items()):
        i +=1
        new_value = {k:v for k,v in value.items() if v >0.5}
        dict_result[key] = new_value
        # if i ==100:
        #     break
    print(dict_result)

def intersection_right50():
    l1 = txt_handle("/home/nhuvn/count_type/output/right50_en.txt")
    l2 = txt_handle("/home/nhuvn/count_type/output/right50_ja.txt")
    l3 = txt_handle("/home/nhuvn/count_type/output/right50_vi.txt")
    l4 = set(l1).intersection(set(l3))
    union = set(l1).union(set(l2)).union(set(l3))
    print(len(union))
    for i in set(union):
        print(i)

# def get_prop():
#     SELECT ?a ?aLabel ?propLabel
# WHERE
# {
#   ?item rdfs:label "Hanoi"@en.
#   ?item ?a wd:27566.

#   SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } 
#   ?prop wikibase:directClaim ?a .
# }
#way 2:
# SELECT ?a ?aLabel ?propLabel
# WHERE
# {
#   wd:1858 ?a wd:27566.

#   SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } 
#   ?prop wikibase:directClaim ?a .
# }
def get_based_number_type_article():
    l = utils.txt_handle("/home/nhuvn/dataset/130K_id.txt")
    for i in l:
        l1 = db.get_instance_of(i)
        # if l1:
        #     print(f"{i},{l1[0]}")
        if l1 and len(l1) ==1:
            print(f"{i},{l1[0]}")
def get_fre50_1():
    dict_t = utils.read_json("/home/nhuvn/dataset/distribution_50_vi.json")
    # i = 0
    dict_value = {}
    dict_result = {}
    l = []
    for key,value in tqdm(dict_t.items()):
        # i +=1
        for k,v in value.items():
            if v>=50:
                dict_value[k] = v
        dict_result[key] = dict_value
        dict_value = {}
        # if i ==3:
        #     break
    for k1,v1 in dict_result.items():
        for k2,v2 in v1.items():
            print(f"{k1},{k2}")
            # k2_new = db.get_label(k2)
            # if k2_new:
            #     print(f"{db.get_label(k1)},{k2_new}")
            # else:
            #     print(f"{db.get_label(k1)},{k2}")
        
def get_intersection():
    dict_vi = utils.read_json("/home/nhuvn/dataset/get_fre50_type_vi.json")
    dict_en = utils.read_json("/home/nhuvn/dataset/get_fre50_type_en.json")
    dict_ja = utils.read_json("/home/nhuvn/dataset/get_fre50_type_ja.json")
    l1 = []
    l2= []
    l_vi = []
    dict_result = {}
    dict_final = {}
    for k1,v1 in dict_en.items():
        for k2,v2 in dict_ja.items():
            if k1 ==k2: 
                l = set(v1).intersection(set(v2))
                dict_result[k1] = l
    for k3,v3 in dict_result.items():
        for k4,v4 in dict_vi.items():
            if k3 == k4:
                l = set(v3).intersection(set(v4))
                dict_final[k3] = l

    for k5,v5 in dict_final.items():
        if v5:
            for i in v5:
                a = db.get_label(i)
                # if a:
                #     l1.append(a)
            # b = ",".join(l1)
                print(f"{db.get_label(k5)},{a}")
            # l1 = []
       
def csv2dict():
    data = defaultdict(list)
    dictfile = defaultdict(list)
    csvFilePath = "/home/nhuvn/dataset/3M.csv"
    with open(csvFilePath,encoding='UTF-8') as csvf:
        #f=io.TextIOWrapper(csvf, encoding='ISO-8859-1')
        csvReader = csv.DictReader(csvf)
        # i = 0
        for rows in tqdm(csvReader):
            key = rows['type']
            if key in data.keys():
                data[key].append(rows['_id'])
            else:
                data[key] = [rows['_id']] 
    utils.write_json(data,"/home/nhuvn/dataset/types_of_3M_en.json")

def select_600type():
    list_type = utils.txt_handle("/home/nhuvn/dataset/600types.txt")
    dict_ent = utils.read_json("/home/nhuvn/dataset/types_of_3M_en.json")
    for type in tqdm(list_type):
        for key,value in dict_ent.items():
            if type == key:
                print(f"{type},{len(value)}")

def diff_list():
    list_type = utils.txt_handle("/home/nhuvn/dataset/600types.txt")
    data3M = pd.read_csv("/home/nhuvn/dataset/3M.csv")
    with gzip.open('/home/nhuvn/dataset/all_no_human_en.pkl.gz', 'rb') as f:
        data = pickle.load(f)
    list_3M = data3M['_id'].tolist()
    set_ent = set(data).difference(set(list_3M))
    dict_id = defaultdict(list)
    # print(len(set_ent))
    for i in tqdm(list_type):
        for j in set_ent:
            a = db.get_instance_of(j)
            if a and (i in a):
                if i in dict_id.keys():
                    dict_id[i].append(j)
                else:
                    dict_id[i] = [j]
            else:
                continue
    with gzip.open('/home/nhuvn/dataset/dict_600type_remain_en.pkl.gz', 'wb') as f:
        pickle.dump(dict_id, f)
def get_title():
    list_1K = utils.txt_handle("/home/nhuvn/vir_model/infer_data/1K_id.txt")
    for i in list_1K:
        title = db.get_labels(i,"en")
        # print(f'{i},{title}')
        print(title)
def create_hlink_type():
    data = []
    l = utils.txt_handle("/home/nhuvn/vir_wd/code/data_rec/119_type.txt")

    for i in tqdm(l):
        l1 = [i,f"{db.get_label(i)}" if db.get_label(i) is not None else i]
        data.append(l1)

    utils.write_csv(data, "/home/nhuvn/vir_wd/code/data_rec/119_type.csv")
def create_article():
    id =0
    data=[]
    l = utils.txt_handle("/home/nhuvn/vir_wd/code/data_rec/119_list_qid.txt")
    for i in tqdm(l):
        l1 = [id,i,f"{db.get_label(i)}" if db.get_label(i) is not None else i,"en"]
        id +=1
        data.append(l1)
    for i in tqdm(l):
        l1 = [id,i,f"{db.get_label(i)}" if db.get_label(i) is not None else i,"ja"]
        id +=1
        data.append(l1)
    for i in tqdm(l):
        l1 = [id,i,f"{db.get_label(i)}" if db.get_label(i) is not None else i,"vi"]
        id +=1
        data.append(l1)
    utils.write_csv(data, "/home/nhuvn/vir_wd/code/data_rec/119_article.csv")
    print(len(data))
def get_qid_article(csvfile):
    dict_Q = defaultdict(list)
    list_Q =[]
    
    with open(csvfile,encoding='UTF-8') as csvf:
        #f=io.TextIOWrapper(csvf, encoding='ISO-8859-1')
        csvReader = csv.DictReader(csvf)
        i =0
        # Q_ID=""
        for row in tqdm(csvReader):
            # print(row)
            key = row['article_id']
            i +=1
            # print(key)
            ID = db.get_qid(row['title'])
            
            if key in dict_Q.keys():
                if ID is not None:
                    dict_Q[key].append(ID)
            else:
                if ID is not None:
                    dict_Q[key] = [ID]
            # if i ==10:
            #     break
    # print(dict_Q)
    print(len(dict_Q))
    # utils.write_json(dict_Q,outfile)
    # with gzip.open('/home/nhuvn/vir_model/infer_data/dict_featured_article1K_hlink_en.pkl.gz', 'wb') as f:
    #     pickle.dump(dict_Q, f)

    #output file is a dict as pkl format:
    with gzip.open('/home/nhuvn/hyperlink1/data/si/pageid_qid_si.pkl.gz', 'wb') as f:
        pickle.dump(dict_Q, f)
def get_type(file)
    dict_result = {}
    for i in file:
        type = db.get_instace_of(Qid)
        if type:
            dict_result[i]= type
    return dict_result 

if __name__ == '__main__':
    # dict_type = utils.read_json("/home/nhuvn/count_type/output/count_relation_ja.json")
    # outfile = "/home/nhuvn/vir_wd/count_type/output/couple_max.csv"
    # txtfile = "/home/nhuvn/count_type/output/type_relation_common_count.txt"
    # outfile1 = "/home/nhuvn/count_type/output/type_relation_common_count_label.txt"
    #get_max_type(dict_type,outfile)
    #test(dict_type)
    get_type50(dict_type)
    # get_type1(txtfile,outfile1)
    # get_len_max(dict_type)
    # commonlist()
    # type_label()
    # compare2()
    # intersection_right50()
    # type_left()
    # get_based_number_type_article()    
    # l = utils.txt_handle("/home/nhuvn/wikidb/t1.txt")
    # get_fre50_1()
    # get_intersection()
    # read_pkl()
    # diff_list()
    # l = txt_handle("/home/nhuvn/dataset/600_2.txt")
    # print(len(l))
    # csv2dict()
    # get_title()
    # select_600type()
    # create_article()
    # create_hlink_type()
    get_type()
    

    print("completion")