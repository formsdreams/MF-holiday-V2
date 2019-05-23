from datetime import datetime, timedelta
from dateutil import parser, tz
from time import time
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from queryrunner_client import Client
from utils.query_utils import query

class GeoUtils(object):
	@staticmethod
	def get_dimcity(city_list=None, verbose=False):
		dim_city_query_str = '''
							 SELECT city_id, city_name, country_name, country_iso2, timezone
							 FROM dwh.dim_city
							 '''
		if city_list is not None:
			dim_city_query_str.append("AND city_id in ({cityIdList})")
			dim_city_query_str.format(cityIdList=",".join(map(str, city_list))) 
		df_dimcity = query(dim_city_query_str, verbose=verbose)
		if verbose:
			print df_dimcity.info()
			print df_dimcity.sample(2)
		return df_dimcity

	@staticmethod
	def get_city2hexcluster_mapping(city_list=None, namespace='hollywood_prod', verbose=False):
		city2hexc_query_str = '''
							  SELECT distinct 
							  cast(city_id AS int) AS city_id, 
							  hex_cluster_id AS hexcluster_id
							  FROM marketplace_forecasting.hex_cluster_mapping
							  WHERE 1=1
							  AND hex_namespace = 'hollywood-prod'
							  AND is_active=true
							  '''
		if city_list is not None:
			city2hexc_query_str.append("AND city_id in ({cityIdList})")
			city2hexc_query_str.format(cityIdList=",".join(map(str, city_list))) 

		df_city2hexc_map = query(city2hexc_query_str)

		if verbose:
			print df_city2hexc_map.info()
			print df_city2hexc_map.sample(2)
		return df_city2hexc_map

	@staticmethod
	def get_hexcluster2hex_mapping(city_list=None, namespace='hollywood_prod', verbose=False):
		hexc2hex_map_query_str = '''
							 	 SELECT
	    					 	 cast(city_id AS int) AS city_id,
	    					 	 hex_cluster_id AS hexcluster_id,
	    					 	 hex_id as origin_hexagon
							 	 FROM marketplace_forecasting.hex_cluster_mapping
							 	 WHERE 1=1
							 	 AND hex_namespace = 'hollywood-prod'
							 	 AND is_active = true 
							 	 '''
		if city_list is not None:
			hexc2hex_map_query_str.append("AND city_id in ({cityIdList})")
			hexc2hex_map_query_str.format(cityIdList=",".join(map(str, city_list))) 
		
		df_hexc2hex_map = query(hexc_map_query_str, verbose=verbose)

		if verbose:
			print df_hexc2hex_map.info()
			print df_hexc2hex_map.sample(2)
		return df_hexc_map


if __name__ == '__main__':
	verbose = True
	df_dimcity = GeoUtils.get_dimcity(verbose=verbose)
	df_hexc_map = GeoUtils.get_hexcluster2hex_mapping(verbose=verbose)



