from utils.geo_utils import GeoUtils
from data_fetcher.golden_dataset_fetcher import GoldenDatasetFetcher
from data_fetcher.holiday_metadata_fetcher import HolidayMetadataFetcher
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

class HolidayFeatureEncoder(object):
	def __init__(self, quantity, namespace='hollywood_prod', city_id_list=None, verbose=False, output_dir=None):
		self.quantity = quantity
		self.city_id_list = city_id_list
		self.df_city2hexc = GeoUtils.get_city2hexcluster_mapping(self.city_id_list, namespace=namespace, verbose=verbose)
		assert (self.df_city2hexc is not None), "The city hexcluster mapping is not correctly queried"
		self.output_dir = output_dir

	@staticmethod
	def lookup_val(val_map, cid, hid, dt, hr):
		if cid not in val_map:
			#print '{} not recorded values'.format(cid)
			return '\N--city_id'
		if hid not in val_map[cid]:
			return '\N--hexc_id'
		dh = (dt + timedelta(hr)).strftime("%Y-%m-%d T %H")
		if dh not in val_map[cid][hid]:
			#print dh
			return '\N--date_hour'
		return val_map[cid][hid][dh]['gd_value']

	def encode_label(self, df_meta, df_activity, verbose=False):
		'''
		# now given the input of metadata and the acitivity, prepare features
		'''
		# query city hexcluster map
		
		holiday_date_hour = pd.DataFrame.from_dict([{'hour': h, 'holiday_date': d} for h in range(24) for d in df_meta.holiday_date.unique()])
		##############################################################################################
		df = df_meta.merge(self.df_city2hexc, on=['city_id'], how='inner').merge(holiday_date_hour, on='holiday_date', how='inner')

		df = df[['city_id', 'hexcluster_id', 'holiday_date', 'holiday_name', 'hour']]
		print type(df_activity)
		print type(df_activity.keys()[0])
		print json.dumps(df_activity, indent=2)[:100]
		df['label'] = df.apply(lambda x: HolidayFeatureEncoder.lookup_val(df_activity, x['city_id'], x['hexcluster_id'], x['holiday_date'], x['hour'])
																		  , axis=1)
		if verbose:
			print df.shape
			print df[~df.label.isin(["\N--city_id"])].shape
			print df[~df.label.isin(["\N--city_id", "\N--hexc_id"])].shape
			print df[~df.label.isin(["\N--city_id", "\N--hexc_id", "\N--date_hour"])].shape
		
		df = df[~df.label.isin(["\N--city_id", "\N--hexc_id", "\N--date_hour"])]
		
		print "...Finish encoding the label "
		
		if verbose:
			print df.holiday_date.min(), df.holiday_date.max()
			print sorted(df.city_id.unique())
			print df.info()
			print df.sample(2)

		return df

	@staticmethod
	def encode_reference_feature(city_hexc, df_meta, lookup_activity, verbose=False):
		reference_date_hour = pd.DataFrame.from_dict([{'hour': h, 
													   'reference_date_id': d} 
													   for h in range(24) 
													   for d in df_meta.reference_date_id.unique()])
		df_meta_indexed = df_meta.merge(city_hexc, on=['city_id'], how='inner').merge(reference_date_hour, 
																							  on='reference_date_id',
																							  how='inner')

		df_ref_days = df_meta_indexed[['city_id', 'hexcluster_id', 
									   'holiday_date', 'reference_date_id', 
									   'holiday_name', 'hour']]
		df_ref_days['reference_val'] = df_ref_days.apply(lambda x: HolidayFeatureEncoder.lookup_val(lookup_activity,
																									x['city_id'], 
																									x['hexcluster_id'], 
																									x['reference_date_id'], 
																									x['hour'])
														 , axis=1)
		if verbose:
			print df_ref_days.shape
			print df_ref_days[~df_ref_days.reference_val.isin(["\N--city_id"])].shape
			print df_ref_days[~df_ref_days.reference_val.isin(["\N--city_id", "\N--hexc_id"])].shape
			print df_ref_days[~df_ref_days.reference_val.isin(["\N--city_id", "\N--hexc_id", "\N--date_hour"])].shape
		
		df_ref_days = df_ref_days[~df_ref_days.reference_val.isin(["\N--city_id", "\N--hexc_id", "\N--date_hour"])]

		if verbose:
			print df_ref_days.holiday_date.min(), df_ref_days.holiday_date.max()
			print sorted(df_ref_days.city_id.unique())
			print df_ref_days.info()
			print df_ref_days.sample(2)

		return df_ref_days


	# get last year value lookup
	def encode_feature_lastYearMatchedVal(self, df_meta, df_activity, verbose=False):
		df_lastYearMeta = df_meta
		print "...Finish encoding the last year matched value feature "
		return HolidayFeatureEncoder.encode_reference_feature(self.df_city2hexc, df_lastYearMeta, df_activity, verbose=verbose)

	def encode_feature_laggedWeek(self, df_meta, lookup_activity, max_lag=6, verbose=False):
		df_laggedWeek = {}
		for lag_ in range(1, max_lag+1):
			df_laggedWeek_lag_ = df_meta.copy()
			df_laggedWeek_lag_['reference_date_id'] = df_laggedWeek_lag_.apply(lambda x: x['holiday_date'] + timedelta(weeks=-lag_),axis=1)
			if verbose:
				print "Lag week of ", lag_
				print df_laggedWeek_lag_.sample(2)
			df_laggedWeek[lag_] = HolidayFeatureEncoder.encode_reference_feature(self.df_city2hexc, df_laggedWeek_lag_, lookup_activity)
			
			print "...Finish generating lagged feature dataset with lag of {} weeks".format(lag_)
		
		if verbose:
			print df_laggedWeek.keys()#, indent=2)[:100]

		return df_laggedWeek

	def encode_feature_calendar(self, df, verbose=False):
		df['dow'] = df.apply(lambda x: x['holiday_date'].dayofweek, axis=1)
		df['year'] = df.apply(lambda x: x['holiday_date'].year, axis=1)
		print "....Finish adding day of week and year"
		if verbose:
			print df.info()
			print df.sample(2)
		return df

	def encode_holiday_feature(self, df_meta, df_activity, verbose=False):
		features = ['last_year', 
			'year_trend',
			'lag_week_1', 'lag_week_2', 'lag_week_3', 'lag_week_4', 'lag_week_5', 'lag_week_6',
			'day_of_week', 
			'hour_of_day'
		]
		df_label = self.encode_label(df_meta, df_activity, verbose=verbose)

		df_lastYearMatchedVal = self.encode_feature_lastYearMatchedVal(df_meta, df_activity, verbose=verbose)

		# encode the lag of week features
		df_laggedWeek = self.encode_feature_laggedWeek(df_meta, df_activity, verbose=verbose)
		
		# merge all continuous features
		label = df_label
		features = {'matched_ref_val': df_lastYearMatchedVal}
		for lag_ in df_laggedWeek:
			features["lag_week_{}".format(lag_)] = df_laggedWeek[lag_]

		df_holiday = label.copy().drop_duplicates()
		select_cols = ['city_id', 'hexcluster_id', 'holiday_date', 'hour']
		for feature_ in features:
			df_feat = features[feature_][select_cols + ['reference_val']].drop_duplicates()
			df_feat[feature_] = df_feat['reference_val']
			print df_holiday.shape
			df_holiday = df_holiday.merge(df_feat[select_cols + [feature_]], 
										  how='inner',
										  on = select_cols)

		# encode the categorical features
		df_holiday = self.encode_feature_calendar(df_holiday, verbose=verbose)
		print ">>>> .... Finish encoding the holiday features ...! "
		
		if verbose:
			print df_holiday.info()
			print df_holiday.sample(2)
		
		return df_holiday


	def save_holiday_feature(self, df):
		savepath = self.output_dir
		savename = "holiday_feature_{}".format(quantity)
		df.to_csv("{}/{}.csv".format(savepath, savename), encoding='utf-8', index=False)
		print "holiday feature saved to directory {} as {}".format(savepath, savename)
		return


if __name__ == '__main__':
	cityIdList = [214, 240, 599, 1005, 1546, 1472]#None #[1]#
	quantity = 'supply'
	verbose = True
	save = True
	output_dir = "holiday_feature_based_model/data"
	holiday_feature_encoder = HolidayFeatureEncoder(quantity, cityIdList,verbose=verbose,output_dir=output_dir)
	meta_fetcher = HolidayMetadataFetcher(cityIdList, verbose=verbose)
	activity_fetcher = GoldenDatasetFetcher(quantity, cityIdList)
	df_meta = meta_fetcher.query_holiday_metadata(verbose=verbose)
	lookup_activity = activity_fetcher.query_golden_dataset(verbose=verbose)

	df_holiday_feature = holiday_feature_encoder.encode_holiday_feature(df_meta, lookup_activity, verbose=verbose)
	if save:
		holiday_feature_encoder.save_holiday_feature(df_holiday_feature)

