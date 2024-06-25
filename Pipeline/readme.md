# Steps to extract hyperlinks and types in multilingual Wikipedia versions 

# 1. Download data dump
- Download the dumps of page, redirect, prop and langlink files
```
https://dumps.wikimedia.org/emwiki/20220120/enwiki-20220120-pages-meta-current1.xml-p1p41242.bz2
https://dumps.wikimedia.org/emwiki/20220120/enwiki-20220120-langlinks.sl.gz
https://dumps.wikimedia.org/enwiki/20220120/enwiki-20220120-page_props.sl.gz
https://dumps.wikimedia.org/enwiki/20220120/enwiki-20220120-redirect.sl.gz


```
# 2. Create index by Wikimapper
- Install Wikimapper
```
pip install wikimapper
```
- Create index by the following command: 


```
$ wikimapper create enwiki-latest --dumpdir data --target data/index_enwiki-latest.db 
```
# 3. Using dump parser to extract Wikipedia page ID, source articles and target articles 

```
python ../Hlink_RS/DumpParser/get_cats_n_links.py

```
# 4. Get Wikidata Qid of Wikipedia page ID

```
python ../Hlink_RS/Pipeline/get_Qid.py

```

# 5. Extract hyperlink types of hyperlinks

- Based on Wikidata-lite 

```
python ../Hlink_RS/Pipeline/get_type.py

```

