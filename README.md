# Hlink_RS
Requirements:
install pandas, surprise

# Instructions of Use
1. Clone github
```
https://github.com/nhunthp/Hlink_RS.git

```
2. Go to clone folder and run hlink_recommendation.py
- There are some actions: statistic, build (build model and evaluation) and rec (recommendation)
- Choose language combination:
- For example:
Single language:
```
python hlink_recommendation.py --languages vi --action statistic 
```
All languages:
```
python hlink_recommendation.py --languages en ja vi --action statistic 
```
3. Change your "data" directory in the following line of code in the "hlink_recommendation.py" file:
```
../Data/'file name.csv'
```

