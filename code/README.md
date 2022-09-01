# Readme

## Executando o preprocessamento

- Movielens One Million: `python preprocessing.py --dataset=ml-1m --n_folds=5 --n_trials=6`

## Executando Search
- Movielens One Million  
  - SVD: `python searches.py --recommender=SVD --dataset=ml-1m`    
- Yahoo Movie  
  - SVD:     
  
## Executando processamento  
- Movielens One Million  
  - SVD: `python processing.py --recommender=SVD --dataset=ml-1m --fold=1 --trial=1`
  
## Executando Decisão  
- Movielens One Million  
  - SVD: `python step6_decision.py --recommender=SVD`
  - 
  
## Executando Metricas  
- Movielens One Million  
  - SVD: `python step7_charts_tables.py -opt=ANALYZE --dataset=ml-1m`
  - 