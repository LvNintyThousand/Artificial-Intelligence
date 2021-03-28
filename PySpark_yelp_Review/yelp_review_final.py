# import pandas as pd 
# import numpy as np 

# yelp_review = pd.read_csv("F:/Baruch College/My courses/2020 Spring/STA 9760/yelp_review.csv")

# text = yelp_review["text"]

# text.to_csv(r'F:/Baruch College/My courses/2020 Spring/STA 9760/yelp_review_revised.csv', header=None, index=None, sep=' ', mode='a')

# # then I change the filename of "yelp_review_revised.csv" from ".csv" to ".txt" for future use. 

import re
import findspark
import os
import time
# import psutil

os.environ["JAVA_HOME"] = "D:/Java/jdk8"
os.environ["SPARK_HOME"] = "D:/Pyspark/spark"

findspark.init()

from pyspark import SparkContext, SparkConf, sql
from pyspark.serializers import MarshalSerializer # SparkContext add serializer = MarshalSerializer()

conf = SparkConf().setAppName("MyApp").set('spark.driver.memory', '8g').set('spark.executor.memory', '8g').set("spark.executor.cores", '8')
sc = SparkContext(conf = conf, serializer = MarshalSerializer())

begin_time = time.clock()
print(begin_time)

doc = sc.textFile("F:/Baruch College/My courses/2020 Spring/STA 9760/midterm/yelp_review_revised.txt")

flattened = doc.filter(lambda line: len(line) > 0).flatMap(lambda line: re.split("W+", line)).coalesce(24, shuffle = True).cache()

kvPairs = flattened.filter(lambda word : len(word) > 3).map(lambda word: (word.lower(), 1))

countsByWord = kvPairs.reduceByKey(lambda v1, v2 : v1+v2).sortByKey(ascending = False)

topWords = countsByWord.map(lambda x: (x[1], x[0])).sortByKey(ascending = False)

print(topWords.take(10))

run_time = time.clock() - begin_time 
print(run_time, "seconds")

# the output is showed below:

# [(6318, 'cons:'), 
#  (4676, 'pros:'), 
#  (3126, 'e will be back!"'),
#  (2563, 'ild '), 
#  (2347, 'e will be back."'), 
#  (2320, 'food:'), 
#  (2196, 'e will definitely be back!"'), 
#  (1828, '"the '), 
#  (1745, 'the '), 
#  (1726, 'enjoy!"')]
