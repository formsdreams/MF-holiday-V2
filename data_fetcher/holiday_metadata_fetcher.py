from utils.query_utils import query
import json
class HolidayMetadataFetcher(object):
	def __init__(self, city_id_list=None, output_dir="./", verbose=False):
		self.city_id_list = city_id_list
		self.output_dir = output_dir
		if verbose:
			print "assigning city id list to ", self.city_id_list
			print "assigning output_dir to", self.output_dir


	def fetch_raw_holiday_info(self, verbose=False):
		query_str_tmpl = '''
		select
		  city_id,
		  holiday_name,
		  date_id as holiday_date,
		  reference_date_id
		FROM
		  marketplace_forecasting.event_truth_holidays
		WHERE
		  1=1
		  AND
		  reference_date_id is not NULL
		'''
		if self.city_id_list is not None:
			query_str_tmpl += "AND city_id in ({cityIdList})"
			result = query(query_str_tmpl.format(cityIdList=','.join(map(str, self.city_id_list))), verbose)
		else:
			result = query(query_str_tmpl)
		return result
		
	def check_reference_days_rules(self, holiday_info, verbose=False):
		select_cols = holiday_info.columns
		# reference day is more than 2 weeks from holiday date
		min_wks_diff = 20
		holiday_info['ref_diff_in_wks'] = holiday_info.apply(lambda x: (x['holiday_date']
																		  -x['reference_date_id']).days/7.0, 
															   axis=1)
		violate = holiday_info.query("ref_diff_in_wks < {}".format(min_wks_diff))
		if not violate.empty:
			print "detect errorous reference day of size {}".format(violate.shape)
		holiday_info = holiday_info.query("ref_diff_in_wks >= {}".format(min_wks_diff))
		
		# check random and test holidays
		holiday_info['is_random_or_test'] = holiday_info.apply(lambda x: 1 
															   if ('random' in x['holiday_name'].lower()) or
															   ('test' in x['holiday_name']) else 0, axis=1)
		violate = holiday_info.query("is_random_or_test == 1")
		if not violate.empty:
			print "detect errorous holiday name of size {}".format(violate.shape)
		holiday_info = holiday_info.query("is_random_or_test == 0")
		return holiday_info[select_cols]

	def query_holiday_metadata(self, verbose=False):
		
		raw_info = self.fetch_raw_holiday_info(verbose)
		if verbose:
			print raw_info.info()
		holiday_metadata = self.check_reference_days_rules(raw_info, verbose)

		return holiday_metadata

	def save_holiday_metadata(self, df):
		df.to_csv("{}/holiday_metadata.csv".format(self.output_dir), encoding='utf-8', index=False)

	def main(self, save=False, verbose=False):
		metadata = self.query_holiday_metadata(verbose)
		if save:
			self.save_holiday_metadata(metadata)

if __name__ == '__main__':
	verbose = True
	cityIdList = [1472,1546]
	outputDir = "./holiday_feature_based_model/data/"
	meta_fetcher = HolidayMetadataFetcher(cityIdList, output_dir = outputDir, verbose=verbose)
	meta_fetcher.main(save=True, verbose=verbose)


