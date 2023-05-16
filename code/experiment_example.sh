python step1_preprocessing.py -opt=SPLIT --dataset=yahoo-movies

python step1_preprocessing.py -opt=DISTRIBUTION --dataset=yahoo-movies --distribution=CWS

python step2_searches.py -opt=RECOMMENDER --recommender=SVD --dataset=yahoo-movies

python step2_searches.py -opt=CONFORMITY --recommender=SVD --dataset=yahoo-movies

python step3_processing.py --dataset=yahoo-movies --recommender=SVD

python step4_postprocessing.py --dataset=yahoo-movies --recommender=SVD --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --distribution=CWS

python step5_metrics.py -opt=RECOMMENDER --dataset=yahoo-movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step5_metrics.py -opt=CONFORMITY --dataset=yahoo-movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step6_protocol.py -opt=RECOMMENDER --dataset=yahoo-movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step6_protocol.py -opt=CONFORMITY --dataset=yahoo-movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE