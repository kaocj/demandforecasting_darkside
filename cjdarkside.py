from google.cloud import bigquery
from datetime import date, timedelta
from pyspark.sql import SparkSession

from  pyspark.sql.functions import abs

from pyspark.sql.functions import monotonically_increasing_id 

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.clustering import KMeans

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import col,sum

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler,ChiSqSelector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

import numpy as np
import pandas
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.sql.functions import when

from pyspark.sql.functions import col,sum

from pyspark.ml.pipeline import PipelineModel

from datetime import date, timedelta

from google.cloud import bigquery

from pyspark.sql.functions import col, when

import pandas as pd

import time
start_time = time.time()

spark = SparkSession.builder.master("local[*]").getOrCreate()
today = date.today()

import warnings
warnings.filterwarnings('ignore')

def holiday(dm):
  if dm=='1-1': return 'NewYear'
  elif dm=='5-2':return 'ChineseDay'
  elif dm=='8-4':return 'Chakri'
  elif dm=='13-4':return 'Songkran'
  elif dm=='14-4':return 'Songkran'
  elif dm=='15-4':return 'Songkran'
  elif dm=='16-4':return 'Songkran'
  elif dm=='1-5':return 'Labour'
  elif dm=='6-5':return 'King'
  elif dm=='9-5':return 'Ploughing'
  elif dm=='3-6':return 'Queen'
  elif dm=='12-8':return 'Mother'
  elif dm=='14-10':return 'King9Passing'
  elif dm=='23-10':return 'Chulalongkorn'
  elif dm=='5-12':return 'King9'
  elif dm=='10-12':return 'Constitution'
  elif dm=='24-12':return 'BeforeChristmas'
  elif dm=='25-12':return 'Christmas'
  elif dm=='30-12':return 'NewYearEve'
  elif dm=='31-12':return 'NewYearEve'
  else: return 'Normal'

def holiday_type(dm):
  if dm=='1-1': return 'National'
  elif dm=='5-2':return 'Regional'
  elif dm=='8-4':return 'National'
  elif dm=='13-4':return 'National'
  elif dm=='14-4':return 'National'
  elif dm=='15-4':return 'National'
  elif dm=='16-4':return 'National'
  elif dm=='1-5':return 'National'
  elif dm=='6-5':return 'National'
  elif dm=='9-5':return 'Goverment'
  elif dm=='3-6':return 'National'
  elif dm=='12-8':return 'Public'
  elif dm=='14-10':return 'National'
  elif dm=='23-10':return 'National'
  elif dm=='5-12':return 'Public'
  elif dm=='10-12':return 'National'
  elif dm=='24-12':return 'Public'
  elif dm=='25-12':return 'Public'
  elif dm=='30-12':return 'National'
  elif dm=='31-12':return 'National'
  else: return 'Normal'
    
def welfare(dm):
  if dm=='2019-5': return 500
  elif dm=='2019-6': return 500
  elif dm=='2020-10': return 500
  elif dm=='2020-11': return 500
  elif dm=='2020-12': return 500
  elif dm=='2021-1': return 500
  else: return 700

def tier_convert(t):
  if t=='T1': return 1
  elif t=='T2': return 2
  elif t=='T3': return 3
  elif t=='T4': return 4
  else: return 5

def query_store_location():
  # Configure the query job.
  client = bigquery.Client()
  job_config = bigquery.QueryJobConfig()
  
  # query location of existing store
  query ='''SELECT code AS BranchCode, lat,lng FROM `thanos-241910.DataMaster.StoreLocation` LIMIT 1000'''
  query_job = client.query(query, job_config=job_config)
  df = query_job.to_dataframe()
  


def query_weather():
  # Configure the query job.
  client = bigquery.Client()
  job_config = bigquery.QueryJobConfig()
  
  # query weather and location of the station in order to map with the store location
  # weather data comes into daily granularity
  # Note : data is lacking for 5 days
  query ='''
      WITH
        weather AS (SELECT
          date,
          temp,
          `max`,
          `min`,
          rain_drizzle,
          wdsp,
          dewp,
          slp,
          visib,
          stn,
          s.name,
          s.lat,
          s.lon
        FROM
          `bigquery-public-data.noaa_gsod.gsod{year}`
        LEFT JOIN
          `bigquery-public-data.noaa_gsod.stations` s
        ON
          stn = usaf
        WHERE
          stn IN (
          SELECT
            DISTINCT usaf
          FROM
            `bigquery-public-data.noaa_gsod.stations`
          WHERE
            country = 'TH') )
      SELECT
        *
      FROM
        weather a
      where 
        a.date=(select MAX(date) from weather b where b.name=a.name)
        '''.format(year = str(today.year))
  query_job = client.query(query, job_config=job_config)
  df = query_job.to_dataframe()
  
  return df

def map_weather_to_store():
  # join the nearest station with the store in order to map the weather to store
  df_2 = query_store_location()
  df_3 = query_weather()
  
  df = df_2.assign(foo=1).merge(df_3.assign(foo=1),on='foo').drop('foo', 1)
  df['dis_store'] = ((df['lat_x']-df['lat_y'])**2 + (df['lng']-df['lon'])**2)**(1/2)
  df['Rank'] = df.groupby(['BranchCode'])['dis_store'].rank(ascending=True)
  #df = df[df['Rank']==1.0]
  df = df[['BranchCode','name','temp']].reset_index(drop=True)
  df = spark.createDataFrame(df)
  
  return df

def one_hot_encoder_promotion(df,_input):
    stringIndexer = StringIndexer(inputCol=_input, outputCol="promotion_idx")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    encoder = OneHotEncoder(dropLast=False, inputCol="promotion_idx", outputCol="promotion_vec")
    encoded = encoder.transform(indexed)
    df = encoded
    return df

def one_hot_encoder_holiday(df,_input):
    stringIndexer = StringIndexer(inputCol=_input, outputCol="holiday_idx")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    encoder = OneHotEncoder(dropLast=False, inputCol="holiday_idx", outputCol="holiday_vec")
    encoded = encoder.transform(indexed)
    df = encoded
    return df

def one_hot_encoder_holidaytype(df,_input):
    stringIndexer = StringIndexer(inputCol=_input, outputCol="holidaytype_idx")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    encoder = OneHotEncoder(dropLast=False, inputCol="holidaytype_idx", outputCol="holidaytype_vec")
    encoded = encoder.transform(indexed)
    df = encoded
    return df

def one_hot_encoder_tier(df,_input):
    stringIndexer = StringIndexer(inputCol=_input, outputCol="tier_idx")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    encoder = OneHotEncoder(dropLast=False, inputCol="tier_idx", outputCol="tier_vec")
    encoded = encoder.transform(indexed)
    df = encoded
    return df

def string_indexer_tier(df,_input):
    indexer = StringIndexer(inputCol="Tier", outputCol="tier_vec")
    df = indexer.fit(df).transform(df)
    return df

def convert_string_to_numeric(df,_input):
    # _input = ['types','Holiday','HolidayType']
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_NUMERIC").fit(df) for column in _input]
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    return df

def encode_holiday(df):
  split_col = F.split(df['SalDate'], '-')

  df = df.withColumn('YYYY', split_col.getItem(0).cast('integer'))\
          .withColumn('MMMM', split_col.getItem(1).cast('integer'))\
          .withColumn('DDDD', split_col.getItem(2).cast('integer'))

  df = df.withColumn('DDDD-MMMM', F.concat(F.col('DDDD'),F.lit('-'), F.col('MMMM')))\
          .withColumn('YYYY-MMMM', F.concat(F.col('YYYY'),F.lit('-'), F.col('MMMM')))
    
  df = df.withColumn('YY', split_col.getItem(0).cast('integer'))
  df = df.withColumn('MM', split_col.getItem(1).cast('integer'))
  
  #print('Size of Original Data Set: ',(df.count(), len(df.columns)))

  holiday_func1 = F.udf(holiday, StringType())
  holiday_func2 = F.udf(holiday_type, StringType())
  
  
  df = df.withColumn("Holiday", holiday_func1("DDDD-MMMM"))\
          .withColumn("HolidayType", holiday_func2("DDDD-MMMM"))
  
  df = df.withColumnRenamed("Branchcode","BranchCode")
  
  df = df.withColumn("BranchCode", df.BranchCode.cast("integer"))\
          .withColumn("MaterialCode", df.MaterialCode.cast("integer"))
  return df

def encode_tier(df):
    
  tier_func1 = F.udf(tier_convert, StringType())

  df = df.withColumn("TierNumeric", tier_func1("Tier"))
  return df

def encode_welfare(df):
  welfare_func1 = F.udf(welfare, StringType())
  df = df.withColumn("Welfare", welfare_func1("YYYY-MMMM"))
  return df

def join_df_with_weather(df,weather):
  return df.join(weather,['Branchcode'], how = 'left_outer')

def transform_ps(df):
  df = df.withColumn("Branch", df.Branch.cast("integer"))\
          .withColumn("MaterialCode", df.MaterialCode.cast("integer"))

  df = df.withColumnRenamed("f0_","Price_Sensitivity")\
          .withColumnRenamed("Branch","BC2")\
          .withColumnRenamed("MaterialCode","MC2")
  return df

def transform_tier(df):
  df = df.withColumn("BranchCode", df.BranchCode.cast("integer"))\
          .withColumn("MaterailCode", df.MaterailCode.cast("integer"))

  df = df.withColumnRenamed("BranchCode","BC3")\
          .withColumnRenamed("MaterailCode","MC3")
  return df

def transform_welfare_flag(df, wf):
	df_welfare_flag = wf.toPandas()
	df_welfare_flag = df_welfare_flag.branchcode.astype(int)
	wf_flag_list = df_welfare_flag.values.tolist()

	matches = df["BranchCode"].isin(wf_flag_list)
	new_df = df.withColumn("welfareFlag", when(matches, "1").otherwise("0"))

	return new_df

def transform_welfare_flag_day(df):
	matches = df["DDDD"].isin([1,2,3,4,5])
	new_df = df.withColumn("welfareFlagDay", when(matches, "1").otherwise("0"))

	return new_df

def join_df_with_ps(df,ps):

  df = df.join(ps, (df.BranchCode == ps.BC2) & (df.MaterialCode == ps.MC2))  

  split_col = F.split(df['SalDate'], '-')

  df = df.withColumn('YY', split_col.getItem(0).cast('integer'))
  df = df.withColumn('MM', split_col.getItem(1).cast('integer'))
  
  df = df.withColumn("BranchCode", df["BranchCode"].cast("integer"))\
          .withColumn("MaterialCode", df["MaterialCode"].cast("integer"))
  return df

def join_df_with_tier(df,tier):
  return df.join(tier, (df.BranchCode == tier.BC3) & (df.MaterialCode == tier.MC3) & (df.YY == tier.CALYEAR) & (df.MM == tier.CALMONTH))
  

def read_file(df,_format):
  return spark.read.format(str(_format)).option("header", "true").load(df)

# dfx = spark.read.format("csv").option("header", "true").load(PS_DATASET_PATH)
def transform_ps(df):
    
    # Branch,MaterialCode,MedianSkuBranch
    df = df.withColumn("Branch", df.Branch.cast("integer"))\
            .withColumn("MaterialCode", df.MaterialCode.cast("integer"))

    df = df.withColumnRenamed("f0_","Price_Sensitivity")\
            .withColumnRenamed("Branch","BC2")\
            .withColumnRenamed("MaterialCode","MC2")
    return df

def join_df_with_ps(df,ps):

    df = df.join(ps, (df.BranchCode == ps.BC2) & (df.MaterialCode == ps.MC2),how='left')  

    split_col = F.split(df['SalDate'], '-')

    df = df.withColumn('YY', split_col.getItem(0).cast('integer'))
    df = df.withColumn('MM', split_col.getItem(1).cast('integer'))
    
    df = df.withColumn("BranchCode", df["BranchCode"].cast("integer"))\
            .withColumn("MaterialCode", df["MaterialCode"].cast("integer"))
    return df
    
def join_df_with_tier(df,tier):
    return df.join(tier, (df.BranchCode == tier.BC3) & (df.MaterialCode == tier.MC3) & \
                   (df.YY == tier.CALYEAR) & (df.MM == tier.CALMONTH),how='left')

def join_df_with_weather(df,weather):
    return df.join(weather,['Branchcode'],how = 'left')

def clean(df):
    #df = df.na.fill('NORMAL',subset=['types'])
    df = df.dropna(how='all')
    df = df.dropna()
    df = df.withColumn("welfareFlag", df["welfareFlag"].cast('integer'))
    df = df.withColumn("welfareFlagDay", df["welfareFlagDay"].cast('integer'))
    df = df.withColumn("supPrice", df["supPrice"].cast('integer'))
    df = df.withColumn("BranchCode", df["BranchCode"].cast("integer"))
    df = df.withColumn("MaterialCode", df["MaterialCode"].cast('integer'))
    df = df.withColumn("TotalQtySale", df["TotalQtySale"].cast('integer'))

    df = df.withColumn("Month", df["Month"].cast('integer'))
    df = df.withColumn("Day", df["Day"].cast('integer'))
    df = df.withColumn("Year", df["Year"].cast('integer'))

    df = df.withColumn("avgPriceDis", df["avgPriceDis"].cast('double'))
    df = df.withColumn("Welfare", df["Welfare"].cast('integer'))
    df = df.withColumn("TierNumeric", df["TierNumeric"].cast('integer'))
    df = df.withColumn("avgPrice", df["avgPrice"].cast('double'))
    df = df.withColumn("Price_Sensitivity", df["Price_Sensitivity"].cast('double'))
    df = df.withColumn("temp", df["temp"].cast('double'))
    df = df.cache()

    return df

def extract_date(df):
    #df = df.na.fill('NORMAL',subset=['types'])
    df = df.dropna(how='all')
    df = (df
    .withColumn('Yearday', F.dayofyear(F.col("SalDate")))
    .withColumn('Month', F.month(F.col('SalDate')))
    .withColumn('DayofWeek', F.dayofweek(F.col('SalDate')))
    .withColumn('DayofMonth', F.dayofmonth(F.col('SalDate')))
    .withColumn('Year', F.year(F.col('SalDate')))
    .withColumn('Quarter', F.quarter(F.col('SalDate')))
    .withColumn('WeekOfYear', F.weekofyear(F.col('SalDate')))
    #.withColumn('Week', F.date_trunc('week',F.col('SalDate')))
    .withColumn('MonthQuarter', F.when((df['Day'] <= 8), 0).otherwise(F.when((df['Day'] <= 16), 1).otherwise(F.when((df['Day'] <= 24), 2)                                 .otherwise(3))))
    )
    split_col = F.split(df['SalDate'], '-')

  	df = df.withColumn('DayofMonth', split_col.getItem(2).cast('integer'))

    df = df.cache()
    return df

def rename(df):
    df = df.withColumnRenamed("DDDD","Day")\
            .withColumnRenamed("MMMM","Month")\
            .withColumnRenamed("YYYY","Year")
    return df

def select_variable(df,select_list):
    df = df.select(select_list)
    return df

# check type
import pandas as pd
pd.set_option('max_colwidth', -1) # to prevent truncating of columns in jupyter

def count_column_types(spark_df):
    """Count number of columns per type"""
    return pd.DataFrame(spark_df.dtypes).groupby(1, as_index=False)[0].agg({'count':'count', 'names': lambda x: " | ".join(set(x))}).rename(columns={1:"type"})

def post_score(df):
  df = df.withColumn("predict_qty", when(col("Day") == 1, col("predict_qty") + col("predict_qty")/2)
                                  .when(col("Day") == 2, col("predict_qty") + col("predict_qty")/2.5)
                                  .when(col("Day") == 3, col("predict_qty") + col("predict_qty")/2.5)
                                  .when(col("Day") == 4, col("predict_qty") + col("predict_qty")/2.5)
                                  .when(col("Day") == 5, col("predict_qty") + col("predict_qty")/2.5)
                                  .otherwise( col("predict_qty") ))
  return df

# EVALUATION

def evalEachClass(df, lower_bound, upper_bound, l, m):
    all_c = df.count()
    df = df.filter( (df.label  >= lower_bound) & (df.label  <= upper_bound)  )
    print("count = ", df.count())
        
    predictionsDF  = m.transform(df)
    predictionsDF.cache()
    result_mae = 0
    if df.count() == 0: result = [0,0,0,0]
    for metricName in ['rmse','mse','r2','mae']:
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="predict_qty", metricName=metricName)
        evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="predict_qty", metricName='mae')
        if(predictionsDF.count() == 0):
            break
        result = evaluator.evaluate(predictionsDF)
        result_mae = evaluator_rmse.evaluate(predictionsDF)
        #print ('%s = %g' % (metricName,result))
        l.append('%s' % (result))
    l.append(df.count())
    print('Evaluate Done: Lower Bound: ',lower_bound,' ,Upper Bound: ',upper_bound, ' with error (mae): ', result_mae)

    
def to_Report1(df, name_csv, list_criterea):
    list_range = ['1','2','3','4 to 6','7 to 10','11 to 15','16 to 25','26 to 40','41 to 60','61 to 100','101 to 200','201 to 400','401 to 700','701 to 1200','1201 to 1800','1801 to 3000']
    d1 = {'evaluations': list_criterea}
    df2 = pd.DataFrame(d1)
    df3 = pd.DataFrame(df2['evaluations'].values.tolist(), columns=['rmse','mse','r2','mae','count'])
    df3['Range'] = pd.DataFrame(list_range)
    df3['Range'] = df3['Range'].astype(object)
    df3['all'] = df.count()
    df3 = df3.set_index(df3['Range'])
    df3 = df3[[ 'mae','rmse', 'mse', 'r2','count','all']]
    df3.to_csv(name_csv)
    print('--- Report1 Complete Done !!')
    print(df3)
    return df3
 
def to_Report2(l, name_csv):    
    new_df = pd.concat(l)
    new_df.to_csv(name_csv)
    print('--- Report2 Complete Done !!')
    print(new_df)
    return new_df

def to_Clear(df, m):
    predictionsDF_All  = m.transform(df)
    predictionsDF_All2 = predictionsDF_All.withColumn( 'prediction_type', F.when( F.col("predict_qty") >= F.col("label"), 'Upper').otherwise('Lower') )
    predictionsDF_All3 = predictionsDF_All2.withColumn('prediction_diff',abs(predictionsDF_All2.predict_qty - predictionsDF_All2.label))
    predictionsDF_All3 = predictionsDF_All3.withColumn("prediction_diff", predictionsDF_All3["prediction_diff"].cast("double"))
    return predictionsDF_All3

    
def evalEachClass2(df, lower, upper, l):
    raw = df
    df2 = df.filter( (df.label  >= lower) & (df.label  <= upper) & (df.prediction_type  == 'Lower') ).groupBy('prediction_type').count().toPandas()
    df = df.filter( (df.label  >= lower) & (df.label  <= upper) & (df.prediction_type  == 'Lower') ).groupBy('prediction_type').mean().toPandas()
    df['Range'] = str(lower)+'-'+str(upper)
    df = df[['Range','prediction_type', 'avg(avgPriceDis)', 'avg(avgPrice)', 'avg(supPrice)', 'avg(predict_qty)', 'avg(prediction_diff)']]
    df['count_of_each_part'] = df2['count'][0] if len(df2) > 0 else 'null'
    df_lower = df
    
    l.append(df_lower)
    
    df = raw
    df2 = df.filter( (df.label  >= lower) & (df.label  <= upper) & (df.prediction_type  == 'Upper')).groupBy('prediction_type').count().toPandas()
    df = df.filter( (df.label  >= lower) & (df.label  <= upper) & (df.prediction_type  == 'Upper') ).groupBy('prediction_type').mean().toPandas()
    df['Range'] = str(lower)+'-'+str(upper)
    df = df[['Range','prediction_type', 'avg(avgPriceDis)', 'avg(avgPrice)', 'avg(supPrice)', 'avg(predict_qty)', 'avg(prediction_diff)']]
    df['count_of_each_part'] = df2['count'][0] if len(df2) > 0 else 'null'
    df_upper = df 
    l.append(df_upper)
    print('Evaluate Done: Lower Bound: ',lower,' ,Upper Bound: ',upper)

