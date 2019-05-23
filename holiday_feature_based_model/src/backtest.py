from utils.geo_utils import GeoUtils
from data_fetcher.golden_dataset_fetcher import GoldenDatasetFetcher
from data_fetcher.holiday_metadata_fetcher import HolidayMetadataFetcher
import pandas as pd
import numpy as np
import os
import sys
import json
from time import time
from datetime import datetime, timedelta
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Backtest(object):
	default_backtest_params = {}
	def __init__(self, quantity, backtest_params=None, data_dir=None, output_dir=None):
		self.quantity = quantity
		self.backtest_params = Backtest.default_backtest_params if backtest_params is None else backtest_params
		self.data_dir = data_dir
		self.output_dir = output_dir

		self.df_backtest = None

	def load_data(self, filename):
		self.df_backtest = pd.read_csv("{}/{}.csv".format(self.data_dir, filename))
		return


	def prepare_data(self):
		self.cat_enc = preprocessing.OneHotEncoder()
		self.cat_enc.fit(self.df_backtest[self.backtest_params['cat_cols']].values)
		self.df_backtest['holiday_date'] = self.df_backtest['holiday_date'].astype(str)
		self.df_backtest['date_rank'] = self.df_backtest['holiday_date'].rank()
		return

	def backtest_step(self, df_train, df_test, feature_cols, label_col, cat_enc):
	    print "Features: ", feature_cols
	    train_feature_matrix = df_train[feature_cols].values
	    train_label_vector = df_train[label_col].values
	    test_feature_matrix = df_test[feature_cols].values
	    test_label_vector = df_test[label_col].values
	    
	    train_feature_matrix = np.concatenate((train_feature_matrix, 
	                                       cat_enc.transform(df_train[self.backtest_params['cat_cols']].values).toarray()), axis=1)

	    test_feature_matrix = np.concatenate((test_feature_matrix, 
	                                           cat_enc.transform(df_test[self.backtest_params['cat_cols']].values).toarray()), axis=1)
	    
	    reg = linear_model.LinearRegression()
	    reg = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=200)
	    reg.fit(train_feature_matrix, train_label_vector)
	    train_pred = reg.predict(train_feature_matrix)
	    test_pred = reg.predict(test_feature_matrix)
	    print "feature importance", reg.feature_importances_
	    print "Training................."
	    print "Mean absolute error: {}".format(mean_squared_error(train_label_vector, train_pred))
	    train_perf = mean_absolute_error(train_label_vector, train_pred) / np.mean(train_label_vector)
	    print "wMAPE: {}".format(train_perf)
	    
	    g_t_train = df_train[['city_id', 'hexcluster_id','holiday_date', 'hour']]
	    g_t_train['prediction'] = train_pred.astype('float')
	    g_t_train['label'] = train_label_vector.astype('float')
	    

	    print "Test................."
	    print "Mean absolute error: {}".format(mean_squared_error(test_label_vector, test_pred))
	    test_perf = mean_absolute_error(test_label_vector, test_pred) / np.mean(test_label_vector)
	    print "wMAPE: {}".format(test_perf)

	    g_t_test = df_test[['city_id', 'hexcluster_id','holiday_date', 'hour']]
	    g_t_test['prediction'] = test_pred.astype('float')
	    g_t_test['label'] = test_label_vector.astype('float')
	    
	    return g_t_train, train_perf, g_t_test, test_perf

	def save_backtest_performance(self):
		t_label = time()
		with open('{}/perf_backtest_{}_{}.json'.format(self.output_dir, self.quantity, t_label), 'w') as f:
			json.dump(self.perf_backtest, f)
		with open('{}/params_backtest_{}_{}.json'.format(self.output_dir, self.quantity, t_label), 'w') as f:
			json.dump(self.backtest_params, f)

	def run(self, if_load=False, load_filename=None):
		if if_load:
			self.load_data(load_filename)
		self.prepare_data()
		citydate = self.df_backtest[["city_id", 'holiday_date','date_rank']].drop_duplicates().reset_index()
		print citydate
		citydate['rank'] = citydate.groupby("city_id")['date_rank'].rank(pct=True)
		self.df_backtest = self.df_backtest.merge(citydate, on=['city_id', 'holiday_date'], how='inner')

		backtest_steps = np.linspace(0.1, 1.0, num=10)
		print backtest_steps
		perf_backtest = {'train':[], 'test':[]}
		df_test = None
		for i, step_ in enumerate(backtest_steps):
		    if step_ == 1.0:
		        break
		    next_step_ = backtest_steps[i+1]
		    print ">>>>> Backtesting step {} <<<<<<".format(i)
		    query_train = 'rank <= {}'.format(step_)
		    citydate_train = citydate.sort_values(by=['city_id', 
		                                              'holiday_date']).query(query_train)
		    print "******training size******", citydate_train.shape
		    df_train_step_ = self.df_backtest.query(query_train)
		    
		    query_test = '(rank <= {}) and (rank > {})'.format(next_step_, step_)
		    citydate_test = citydate.sort_values(by=['city_id', 
		                                             'holiday_date']).query(query_test)
		    print "******test size******", citydate_test.shape
		    df_test_step_ = self.df_backtest.query(query_test)
		    df_train_w_pred, wmape_train, df_test_w_pred, wmape_test = self.backtest_step(df_train_step_, 
		                                                                        	 	  df_test_step_, 
		                                                                        	 	  self.backtest_params['feature_cols'], 
		                                                                        	 	  self.backtest_params['label_col'], 
		                                                                        	 	  self.cat_enc)
		    df_test_w_pred['backtest_step'] = i
		    if df_test is None:
		        df_test = df_test_w_pred
		    else:
		        df_test = df_test.append(df_test_w_pred[df_test.columns])
		    perf_backtest['train'].append(wmape_train)
		    perf_backtest['test'].append(wmape_test)
		print "test wmape **** train wmape"
		print np.mean(perf_backtest['test']), np.mean(perf_backtest['train'])
		self.perf_backtest = perf_backtest


if __name__ == '__main__':

	pyparser = argparse.ArgumentParser()
	pyparser.add_argument("quantity", help="parse the golden dataset quantity you want to backtest on", type=str)
	args = pyparser.parse_args()

	backtest_params = {
		'feature_cols': ['lag_week_{}'.format(i) for i in range(1, 7)] + ['matched_ref_val'],
		'label_col': ['label'],
		'cat_cols': ['dow', 'hour']
		}
	quantity = args.quantity
	data_filename = "holiday_feature_{}".format(quantity)
	backtester = Backtest(quantity, backtest_params, 'holiday_feature_based_model/data', 'holiday_feature_based_model/reports')
	backtester.run(True, data_filename)
	backtester.save_backtest_performance()
