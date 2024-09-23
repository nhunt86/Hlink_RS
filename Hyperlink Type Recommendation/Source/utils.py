import csv
import os
import json
import pickle
import gzip
 
def make_dirs(dirname):
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def read_json(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        return json.load(f)


def write_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            f.write(linesep)
            f.flush()


def write_json_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    json_lines = (json.dumps(line, ensure_ascii=False) for line in lines)
    write_lines(json_lines, filename=filename, linesep=linesep, encoding=encoding)

def write_json(obj, filename, indent=3, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(filename, mode="w", encoding=encoding) as f:
        json.dump(obj, fp=f, ensure_ascii=False, indent=indent)

def write_json_gzip(obj, filename, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with gzip.open(filename, mode="wt", encoding=encoding) as f:
        json.dump(obj, fp=f, ensure_ascii=False)

# with gzip.open(jsonfilename, 'rt', encoding='UTF-8') as zipfile:
#     my_object = json.load(zipfile)
def write_jsonl_format_without_key(data, filename, indent=None, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(filename, 'w', encoding=encoding) as f:
        print ("Saving {}".format(filename))
        json.dump(data, f)


def write_jsonl_format(data, filename, indent=None, encoding="ISO-8859-1"):
    make_dirs(os.path.dirname(filename))
    with open(filename, 'w', encoding=encoding) as f:
        print ("Saving {}".format(filename))
        json.dump({'data': data}, f)


def read_json_with_key(fn, key="id", encoding="UTF-8"):
    """
        - each line is a json
        - without indent
    """
    dic = {}
    with open(fn, "r", encoding=encoding) as f:
        for line in f:
            obj = json.loads(line)
            dic[obj[key]] = obj
    return dic


def convert_list2dict(data):
    dict_ = {}
    for item in data:
        key = list(item.keys())[0]
        dict_[key] = item[key]
    return dict_

def read_csv_return_dict(filename):
    dict_re = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dict_re[row[0]] = row[1]
    return dict_re

def pkl_write(data,path):
    with gzip.open(path,'wb') as pkl_file:
        pickle.dump(data,pkl_file)

def pkl_read(path):
    with gzip.open(path,'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def write_txt(data,outfile):
    with open(outfile, 'w') as f:
        for item in data:
            (f.write("%s\n" % item))


def write_csv(data,filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)