import csv
import os
#import exercise3


for log in os.listdir('logs'):
    if 'log' in log:
        f = open('logs/'+log, 'rw')
        reader = csv.reader(f)
        configuration = dict()
        for hp in reader:
            if len(hp) == 2:
                configuration[hp[0]] = float(hp[1])
            else: break
            # run configuration
            print(configuration)
        f.close()
