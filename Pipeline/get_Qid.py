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
database_en = "/home/nhuvn/hyperlink1/database/index_enwiki-20220120.db" 
database_ja = "/home/nhuvn/hyperlink1/database/index_jawiki-20220120.db"
database_vi = "/home/nhuvn/hyperlink1/database/index_viwiki-20220120.db"
database_si= '/home/nhuvn/hyperlink1/data/si24/index_siwiki-latest.db'
mapper_ja = WikiMapper(database_ja)
mapper_vi = WikiMapper(database_vi)
mapper_en = WikiMapper(database_en)
mapper_si = WikiMapper(database_si)

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


def get_ID_si(title):
    ID = None
    title = standard_ja(title)
    if title.startswith("ප්‍රවර්ගය"):
        print("no Q")
    elif title.startswith("File:"):
        print("no Q")
    elif title.startswith("Category:"):
        print("no Q")
    elif title.startswith("ගොනුව:"):
        print("no Q")
    elif title.startswith("රූපය:"):
        print("no Q")
    elif title=="":
        print("test error here....")
    else:
        ID = mapper_si.title_to_id(title)
        
    return ID

def get_hlink_qid():
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
    print(len(dict_Q))
    # utils.write_json(dict_Q,outfile)
    # with gzip.open('/home/nhuvn/vir_model/infer_data/dict_featured_article1K_hlink_en.pkl.gz', 'wb') as f:
    #     pickle.dump(dict_Q, f)

    #output file is a dict as pkl format:
    with gzip.open('/home/nhuvn/hyperlink1/data/si/pageid_hlink_qid_si.pkl.gz', 'wb') as f:
        pickle.dump(dict_Q, f)


def get_qid_article():
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
            ID = get_ID_en(row['title'])
            
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

def pkl_read(path):
    with gzip.open(path,'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def qid_hlink_qid():
    article = pkl_read('/home/nhuvn/hyperlink1/data/si/pageid_qid_si.pkl.gz')
    hlink = pkl_read('/home/nhuvn/hyperlink1/data/si/pageid_hlink_qid_si.pkl.gz')
    list_qid = func.txt_handle('/home/nhuvn/hyperlink1/data/en/q_all.txt')
    dict_result = {}
    count = 0
    for key,value in tqdm(article.items()):
        for k,v in hlink.items():
            if key ==k:
                # print(key,k)
                dict_result[value[0]]= v
    print(len(dict_result))
    # k = 0
    for k1,v1 in dict_result.items():
        if k1 in list_qid:
            count+=1
            print(k1)
    print(count)
    #     print(k1,v1)
    #     k +=1
    #     if k ==2:
    #         break
    # s
def count_hlink():
        hlink = pkl_read('/home/nhuvn/hyperlink1/data/si/pageid_hlink_qid_si.pkl.gz')
        l = []
        for k, v in hlink.items():
            for i in v:
                if i not in l:
                    l.append(i)
        print(len(l))

if __name__ == '__main__':
    # csvfile = "/home/nhuvn/vir_wd/extract_hlink/data/dl/20190120/view_1K.csv"
    # outfile = "/home/nhuvn/vir_wd/data/list_hlink_qitem/key_list_hlink_Q_vi.json"
    #csvfile = '/home/nhuvn/hyperlink1/data/si/articleLink.csv'
    csvfile = '/home/nhuvn/hyperlink1/data/si/article.csv'
    #outfile = ''
    #get Qid of hlinks
    #get_hlink_qid()
    # get_qid_article()
    #si: 7K articles have Qids and hyperlink Qids, 1415: #articles in si
    # qid_hlink_qid()
    count_hlink()
    
    print('completion')
