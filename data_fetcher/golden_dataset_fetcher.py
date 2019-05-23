from utils.query_utils import query
from utils.geo_utils import GeoUtils
from utils.time_utils import TimeUtils
import json
import pandas as pd
import numpy as np

class GoldenDatasetFetcher(object):
	def __init__(self, quantity, city_id_list=None, output_dir=None, verbose=False):
		self.quantity = quantity
		self.city_id_list = city_id_list
		self.output_dir = output_dir

	def fetch_raw_golden_dataset(self, verbose=False):
		query_str_tmpl = '''
						 select datestr, city_id, hexcluster_id, utc_timestamp, 
						 imputed_value, real_value, is_blacklisted, is_holiday
						 from  marketplace_forecasting.forecasting_golden_data_hexc_{quantity}
						 '''
		if self.city_id_list is not None:
			query_str_tmpl += "WHERE city_id in ({cityIdList})"
			query_str = query_str_tmpl.format(quantity=self.quantity, cityIdList=",".join(map(lambda xx: "\'{}\'".format(xx), self.city_id_list))) 
		else:
			query_str = query_str_tmpl.format(quantity=self.quantity)
		result = query(query_str, verbose)
		print result.sample(2)
		result['city_id'] = result['city_id'].astype('int64')
		if verbose:
			print "result of data query for golden dataset>>>"
			print result.info()
			print result.sample(2)
		return result

	def reformat_dataset(self, df_raw, verbose=False):
		df_gd = df_raw
		# add the timezone
		# get dimcity table
		df_dimcity = GeoUtils.get_dimcity(verbose=verbose)
		tz_lookup = {int(x['city_id']): x['timezone'] for x in df_dimcity.to_dict('records')}
		df_gd['timezone'] = df_gd.apply(lambda x: tz_lookup[x['city_id']], axis=1)
		# do time local conversion from timezone column
		df_gd['local_dt_str'] = df_gd.apply(lambda x: TimeUtils.utc_ms_to_local_datestr(x['utc_timestamp'], x['timezone']), axis=1)
		df_gd['gd_value'] = df_gd.apply(lambda x: x['imputed_value'] if x['is_blacklisted'] else x['real_value'], axis=1)
		df_gd = df_gd.groupby(['city_id', 'hexcluster_id', 'local_dt_str','is_holiday'])['gd_value'].sum().reset_index()

		if verbose:
			print "Result of reformating>>>"
			#print "not implemented yet! : ("
			print df_gd.info()
			print df_gd.sample(2)

		return df_gd

	@staticmethod
	def gen_value_lookup(df, value_cols):
		value_map = {}
		for x in df.to_dict('records'):
			cid = x['city_id']
			hid = x['hexcluster_id']
			did = x['local_dt_str']
			if cid not in value_map:
				value_map[cid] = {}
			if hid not in value_map[cid]:
				value_map[cid][hid] = {}
			value_map[cid][hid][did] = {vc: x[vc] for vc in value_cols} 
		return value_map

	def generate_val_lookup_table(self, df_gd, verbose=False):
		val_lookup_table = {'mock_key': 'mock_value'}
		val_lookup_table = GoldenDatasetFetcher.gen_value_lookup(df_gd, ['gd_value', 'is_holiday'])
		if verbose:
			print json.dumps(val_lookup_table, indent=2)[:500]

		return val_lookup_table

	def query_golden_dataset(self, verbose=False):
		df_raw = self.fetch_raw_golden_dataset(verbose=verbose)
		df_gd = self.reformat_dataset(df_raw, verbose=verbose)
		val_lookup_gd = self.generate_val_lookup_table(df_gd, verbose=verbose)
		return val_lookup_gd

	def save_golden_dataset(self, val_map):
		with open('{}/gd_val_lookup_{}.json'.format(self.output_dir, self.quantity), 'w') as f:
			json.dump(val_map,f)

	def main(self, save=False, verbose=False):
		golden_dataset = self.query_golden_dataset(verbose)
		if save:
			self.save_golden_dataset(golden_dataset)


if __name__ == '__main__':
	city_list = [1546, 1472]
	quantity = 'supply'
	verbose = True
	gd_fetcher = GoldenDatasetFetcher(quantity, city_list, output_dir="holiday_feature_based_model/data")
	val_lookup = gd_fetcher.main(save=True, verbose=verbose)
	
