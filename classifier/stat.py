import statistics
import json
from os import listdir
from os.path import isfile, join
import datetime

evaluation_set = './experimental_data/'

def get_execution_time(path):
    files = [ f for f in listdir(path) if isfile(join(path, f)) and ".json" in f ]
    timings_ff = []
    timings_chrome = []
    timings = []
    for f in files:
        with open(path+f,'r') as file:
            data = json.load(file)
        timings.append(round(data['time']/1000))
        if 'Firefox' in data['user_agent']:
            timings_ff.append(round(data['time']/1000))
        elif 'Chrome' in data['user_agent']:
            timings_chrome.append(round(data['time']/1000))


    print('Average execution time: ', "{:0>8}".format(str(datetime.timedelta(seconds=statistics.mean(timings)))), 'Standard deviation: ', "{:0>8}".format(str(datetime.timedelta(seconds=statistics.stdev(timings)))))
    print('Average execution time on Firefox: ', "{:0>8}".format(str(datetime.timedelta(seconds=statistics.mean(timings_ff)))), 'Standard deviation: ', "{:0>8}".format(str(datetime.timedelta(seconds=statistics.stdev(timings_ff)))))
    print('Average execution time on Chrome: ', "{:0>8}".format(str(datetime.timedelta(seconds=statistics.mean(timings_chrome)))), 'Standard deviation: ', "{:0>8}".format(str(datetime.timedelta(seconds=statistics.stdev(timings_chrome)))))


if __name__ == '__main__':
    get_execution_time(evaluation_set)
