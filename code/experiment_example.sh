python step1_preprocessing.py -opt=SPLIT --dataset=twitter_movies

python step1_preprocessing.py -opt=DISTRIBUTION --dataset=twitter_movies --distribution=CWS

python step2_searches.py -opt=RECOMMENDER --recommender=SVD --dataset=twitter_movies

python step2_searches.py -opt=CONFORMITY --recommender=SVD --dataset=twitter_movies

python step3_processing.py --dataset=twitter_movies --recommender=SVD

python step4_postprocessing.py --dataset=twitter_movies --recommender=SVD --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --distribution=CWS

python step5_metrics.py -opt=RECOMMENDER --dataset=twitter_movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step5_metrics.py -opt=CONFORMITY --dataset=twitter_movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step6_protocol.py -opt=RECOMMENDER --dataset=twitter_movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step6_protocol.py -opt=CONFORMITY --dataset=twitter_movies --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step1_preprocessing.py -opt=SPLIT --dataset=ml-1m

python step1_preprocessing.py -opt=DISTRIBUTION --dataset=ml-1m --distribution=CWS

python step2_searches.py -opt=RECOMMENDER --recommender=SVD --dataset=ml-1m

python step2_searches.py -opt=CONFORMITY --recommender=SVD --dataset=ml-1m

python step3_processing.py --dataset=ml-1m --recommender=SVD

python step4_postprocessing.py --dataset=ml-1m --recommender=SVD --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --distribution=CWS

python step5_metrics.py -opt=RECOMMENDER --dataset=ml-1m --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step5_metrics.py -opt=CONFORMITY --dataset=ml-1m --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step6_protocol.py -opt=RECOMMENDER --dataset=ml-1m --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE

python step6_protocol.py -opt=CONFORMITY --dataset=ml-1m --recommender=SVD --distribution=CWS --relevance=NDCG --calibration=VICIS_EMANON2 --tradeoff=LIN --selector=SURROGATE