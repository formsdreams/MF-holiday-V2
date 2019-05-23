from datetime import datetime, timedelta
from dateutil import parser,tz
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

class TimeUtils(object):
	@staticmethod
	def utc_ms_to_local_datestr(utcms, timezone):
	    from_tz = tz.gettz('UTC')
	    to_tz = tz.gettz(timezone)
	    utc_dt = datetime.utcfromtimestamp(utcms)
	    utc_dt = utc_dt.replace(tzinfo=from_tz)
	    local_dt = utc_dt.astimezone(to_tz)
	    local_dt_str = local_dt.strftime("%Y-%m-%d T %H")
	    return local_dt_str