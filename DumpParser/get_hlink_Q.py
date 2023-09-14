import pandas as pd
import csv
from collections import defaultdict
from tqdm import tqdm
from wikimapper import WikiMapper
import numpy as np
from multiprocessing import Process
import utils
from tqdm import tqdm
import gzip, pickle
import functions as func
# database_en = "/home/nhuvn/hyperlink1/database/index_enwiki-20220120.db" 
database_ja = "/home/nhuvn/hyperlink1/database/index_jawiki-20220120.db"
# database_vi = "/home/nhuvn/hyperlink1/database/index_viwiki-20220120.db"
mapper_ja = WikiMapper(database_ja)
# mapper_vi = WikiMapper(database_vi)
# mapper_en = WikiMapper(database_en)

def standard(temp):
    #title =""
    title = str(temp)
    title = title.replace('"','')
    if len(title)>1:
        title= title[0].upper() + title[1:].replace(" ", "_")
    elif len(title)==1:
        title= title[0].upper()
    else:
        title ='099z'
    return title

def standard_ja(temp):
    #title =""
    title = str(temp)
    title = title.replace('"','')
    
    # title = title.replace("('",'').replace("',)",'')
    # title = title.replace(" ", "_")
    return title

def get_ID_en(title):
    ID =  None
    title = standard(title)
    
    if title.startswith("Category:"):
        #ID =None
        print("no Q")
    elif title.startswith("File:"):
        # ID= None
        print("no Q")
    elif title=="":
            print("test error here....")
    else:
            #title = standard(list_title[i])
        ID = mapper_en.title_to_id(title)
    return ID

    
        
def get_ID_vi(title):
    ID = None
    title = standard(title)
       
    if title.startswith("Thể loại:"):
        print("no Q")
    elif title.startswith("File:"):
        print("no Q")
    elif title.startswith("Category:"):
        print("no Q")
    elif title.startswith("Tập tin:"):
        print("no Q")
    # elif title.startswith(":"):
    #     print("no Q")
    elif title=="":
        print("test error here....")
    else:
        ID = mapper_vi.title_to_id(title)
         
    return ID


def get_ID_ja(title):
    ID = None
    title = standard(title)
    if title.startswith("カテゴリー"):
        print("no Q")
    elif title.startswith("File:"):
        print("no Q")
    elif title.startswith("Category:"):
        print("no Q")
    elif title.startswith("ファイル"):
        print("no Q")
    elif title=="":
        print("test error here....")
    else:
        ID = mapper_ja.title_to_id(title)
        
    return ID

def get_data():
    dict_Q = defaultdict(list)
    list_Q =[]
    
    with open(csvfile,encoding='UTF-8') as csvf:
        #f=io.TextIOWrapper(csvf, encoding='ISO-8859-1')
        csvReader = csv.DictReader(csvf)
        i =0
        # Q_ID=""
        for row in tqdm(csvReader):
            # print(row)
            key = row['pageid']
            i +=1
            # print(key)
            ID = get_ID_en(row['target_article_name'])
            
            if key in dict_Q.keys():
                if ID is not None:
                    dict_Q[key].append(ID)
            else:
                if ID is not None:
                    dict_Q[key] = [ID]
            # if i ==10:
            #     break
    # print(dict_Q)
    # print(len(dict_Q))
    # utils.write_json(dict_Q,outfile)
    with gzip.open('/home/nhuvn/vir_model/infer_data/dict_featured_article1K_hlink_en.pkl.gz', 'wb') as f:
        pickle.dump(dict_Q, f)
# def find_item():
#     wik= utils.read_json("/home/nhuvn/vir_wd/data/list_hlink_qitem/key_list_q_all.json")
#     for key in wik.keys():
#         if key == '56667':
#             print (len(wik[key]))


if __name__ == '__main__':
    # csvfile = "/home/nhuvn/vir_wd/extract_hlink/data/dl/20190120/view_1K.csv"
    # outfile = "/home/nhuvn/vir_wd/data/list_hlink_qitem/key_list_hlink_Q_vi.json"
    # get_data()
    # name = func.txt_handle("/home/nhuvn/vir_wd/code/FA_ja.txt")
    # for i in name:
    #     ID = get_ID_ja(i)
    #     if ID:
    #         print(ID)
    #     else:
    #         print("no")
    all_qid = func.txt_handle("/home/nhuvn/hyperlink1/Database/all_qid.txt")
    list_ja = func.txt_handle("/home/nhuvn/vir_wd/code/FA_ja_q.txt")
    for i in list_ja:
        if i in all_qid:
            print(i)
        else:
            continue
    #find_item()
    # with gzip.open('/home/nhuvn/dataset/dict_human_hlink_en.pkl.gz', 'rb') as f:
    #     dict_ent = pickle.load(f)
    # i = 0
    # for k,v in dict_ent.items():
    #     print(k,v)
    #     i +=1
    #     if i ==3:
    #         break
    # print(len(dict_ent))
    print('completion')
