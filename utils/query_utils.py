from datetime import datetime, timedelta
from dateutil import parser
from time import time
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import queryrunner_client
from queryrunner_client import Client, start_adhoc

def query(query, verbose=False):
	start_adhoc()
	qr = Client(user_email='xiaochen.zhang@uber.com')
	try:
		cursor = qr.execute('presto', query)
	except queryrunner_client.ServiceError as e:
		print str(e)
		return
	if verbose:
		print "The query: ", query
		print "The execution id: ", cursor
	result = pd.DataFrame.from_dict(cursor.load_data())
	return result


