# from datetime import datetime
# import json
# import os
import sqlite3
import sys,csv,pickle
from collections import Counter
# import uuid
# import random
import pandas as pd
from time import gmtime, strftime
import utils
import re
import numpy as np
import gzip
# from functoools import lru_cache
import tqdm
from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api
from multiprocessing import Process
from collections import defaultdict

# get all types of a Wikidata id
def get_instansof_values(qid, wikidata_items):
    results = []
    if qid in wikidata_items.keys():
        relations = wikidata_items[qid]['relations']
        for item in relations:
            if item[0] == "P31":
                results.append(item[1])
    return results


def get_entity(entity_id):
    try:
        entity = get_entity_dict_from_api(entity_id)
    except:
        return {
            "id": "",
            "label": "",
            "desc": "",
            "aliases": (),
            "statements": (),
            "enwiki_url": ""
            }
    entity = WikidataItem(entity)
    enwiki = entity.get_sitelinks().get("enwiki")
    enwiki_url = enwiki and enwiki["url"]
    #
    statements = [] # list of relations
    #
    # for statement_group in entity.get_truthy_claim_groups().values():
    for statement_group in entity.get_claim_groups().values(): 
        for statement in statement_group:
            if statement.mainsnak.value_datatype == "wikibase-entityid":
                statements.append((statement.mainsnak.property_id, statement.mainsnak.datavalue.value["id"]))
    # 
    return {
        "id": entity.entity_id,
        "label": entity.get_label(),
        "desc": entity.get_description(),
        "aliases": tuple(entity.get_aliases()),
        "statements": tuple(statements),
        "enwiki_url": enwiki_url
    }


# get label of a Wikidata id 
def get_labelof_values(qid, wikidata_items):
    if qid in wikidata_items.keys():
        entity_e_info = wikidata_items[qid]
    else:
        entity_e_info = get_entity(qid) 
    #
    return entity_e_info['label']


def sort_dict(d): # d is a dict
    import operator
    print('Original dictionary : ',d)
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    print('Dictionary in ascending order by value : ',sorted_d)
    sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    print('Dictionary in descending order by value : ',sorted_d)


def write_txtfile(list_data,outfile):
    with open(outfile, 'w') as f:
        for item in list_data:
        #for item in result:
            (f.write("%s\n" % item))

def txt_handle(tfile):
    my_file = open(tfile, "r")
    content = my_file.read()
    type_all = content.split("\n")
    #print(type_all)
    return type_all 

def Convert(tup, di):
    di = dict(tup)
    return di

def bayes_prob(data): #data is a dict
    posterior = {}
    hypos1 = data.keys()
    if len(data) >0:
        prob = 1/len(data)
        prior1 = pd.Series(prob,hypos1)
        likelihood = tuple(data.values())
        unnorm = prior1*likelihood
        prob_data = unnorm.sum()
        posterior = dict(unnorm / prob_data)
        return posterior #this is a dict
    else:
        return posterior

def dict2table():
    dict_type = utils.read_json(outfile6)
    conn = sqlite3.connect(path_to_db, isolation_level="EXCLUSIVE")
    with conn:
        conn.execute(
            """Create table type_50 (
                left text,
                right text)"""
        )
    c = conn.cursor()
    for key,value in tqdm(dict_type.items()):
        for k,v in value.items():
            c.execute(
                    # "INSERT INTO hlink_en (source_article_id,target_article_name) VALUES (?,?)",
                    "INSERT INTO type_50 (left,right) VALUES (?,?)",
                    (key,k),)
    conn.commit()
    conn.close()

def dif_dict():
    d1 = {"a":1,"b":2,"c":2}
    d2 = {"a":1,"c":2,"e":3}
    dif = set(d2.items())^set(d1.items())
    print(dif)
    union = d1 | d2# merge 2 dictionaries
    merge_sum = dict(Counter(d1)+Counter(d2))
    print(union)

def pkl_write(data,path):
    with open(path,'wb') as pkl_file:
        pickle.dump(data,pkl_file)

def pkl_read(path):
    with open(path,'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
def array2metric():
    a=np.array([1,2,3,4,5])
    b=np.array([9,8,7,6,5])
    c=np.column_stack((a,b)).T
    # d=np.array([2,2,2,2,2])
    # c=np.column_stack((c.T,d)).T
    return c
def load_pkl(path):
    # path = '/home/nhuvn/dataset/1K_feature.pkl'
    f = gzip.open(path,'rb')
    matrix = pickle.load(f)
    f.close()
    # print(type(matrix))
    # print(matrix)
    return matrix

def common_list_of_list(l1): # l1 is a list contains lists [list1,list2,....,listn]
    result = set(l1[0])  
    for s in l1[1:]:
        result.intersection_update(s)
    # print (result)
    return result

def tip_trick(list1):
    dict_counter = dict(Counter(list1)) #count duplicate element in list and return a dict
    {key:val for key, val in dict_counter.items() if val >= 30} #remove element in dict by value 