# Hlink_RS
Requirements:
install pandas, surprise

# Instructions of Use
1. Clone github
```
https://github.com/nhunthp/Hlink_RS.git

```
2. Go to clone folder
```
..Hyperlink Type Recommendation/Source'
```
 and run the file hlink_recommendation.py
- Change your "data" directory in the following line of code in the "hlink_recommendation.py" file:
```
../Data/Rich_case/'file name.csv'
```
- There are some actions: statistic, build (build model and evaluation) and rec (recommendation)
- Choose language combination: en: English, ja: Japanese, vi: Vietnamese
- For example:

Single language: 
```
python hlink_recommendation.py --languages vi --action statistic 
python hlink_recommendation.py --languages vi --action build
python hlink_recommendation.py --languages vi --action rec  
```
All languages:
```
python hlink_recommendation.py --languages en ja vi --action statistic 
python hlink_recommendation.py --languages en ja vi --action build
python hlink_recommendation.py --languages en ja vi --action rec 
```

# Development

The required dependencies are managed by pip. A virtual environment containing all needed packages for development and production can be created and activated by
```
- virtualenv --system-site-packages -p python3
- go to the folder (venv)
- source bin/active
```