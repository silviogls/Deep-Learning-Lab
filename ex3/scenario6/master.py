import numpy as np
import os
import csv
from datetime import datetime
import time


class HPConfiguration:
    def __init__(self, hps_init, mode = 'random'): 
        self.hps_init = hps_init
        self.hps = []
        
        if mode == 'random':
            self.get_next_setting = self.get_next_setting_random
        elif mode == 'grid':
            for hp in hps_init:
                self.hps.append((hp[0], self.get_hp_gen(hp[1],hp[2],hp[3])))
            self.get_next_setting = self.get_next_setting_grid
            
        self.n_hps = len(self.hps_init)
        self.setting = [(hp[0],hp[1].next()) for hp in self.hps]
        self.results = []
        self.current_hp_idx = 0
        self.current_best = [None, 0]
        self.has_more = 0
        
    def get_hp_gen(self, start, stop, step):
        for i in np.arange(start, stop+step, step):
            yield(i)
    
    ## returns True if there are ore unexplored settings
    def has_more_settings(self):
        return self.has_more < self.n_hps
    
    ## returns random settings, never ending
    def get_next_setting_random(self):
        ret = []
        for hp in self.hps_init:
            val = hp[1]+np.random.rand()*(hp[2]-hp[1])
            if 'batch_size' in hp[0] or 'filters' in hp[0]:
                val = int(val)
            ret.append([hp[0], val])
        return ret
    
    ## returns settings using grid search, until there is no more
    def get_next_setting_grid(self):
        ret = list(self.setting)
        try:
            newval = self.hps[self.current_hp_idx][1].next()
        except StopIteration:
            self.has_more += 1
            newval = self.setting[self.current_hp_idx][1]
            
        self.setting[self.current_hp_idx] = (self.hps[self.current_hp_idx][0], newval)
        self.current_hp_idx = (self.current_hp_idx + 1) % len(self.hps)
        return ret
        

## we are going to put ONE setting in each file (makes it simpler to code)
#~ logsdir = "logs"+str(datetime.today()).replace(" ","_").replace(":","_").replace(".","_")+"/"
logsdir = "logs"+"/"

## make the directory for the logs, and delete all the files inside it before running
if not os.path.exists(logsdir):
    os.makedirs(logsdir)
for l in os.listdir(logsdir):
    os.remove(logsdir+l)
    
COUNT = 0
N_WORKERS = 2
hpconf = HPConfiguration([
                    ('nfilters', 10, 100, 10), 
                    ('LR', 0.01, 0.4, 0.02), 
                    ('M', 0, 1, 1e-1), 
                    ('batch_size_train', 32, 256, 32)
                    ], mode='random')

## make initial setting files
for i in range(N_WORKERS):
    f = open(logsdir+'log'+str(COUNT), 'a'); COUNT += 1
    conf = hpconf.get_next_setting()
    print("Writing new setting:")
    print(str(conf)+"\n")
    for hp in conf:
        f.write(str(hp[0])+", "+str(hp[1])+"\n")
    f.write("\n")
    f.close()

current_best = [None, 0]
used_settings = []

while(hpconf.has_more_settings()):
    time.sleep(0.05) ## cpu rest
    ## read returned performances and make new config files
    for logfile in os.listdir(logsdir):
        if 'log' in logfile and '.done' in logfile and '.seen' not in logfile:
            ## read performance
            f = open(logsdir+logfile, "r")
            reader = csv.reader(f)
            conf = []
            for line in reader:
                if len(line) == 2:
                    if 'best validation accuracy' in line[0]:
                        val_acc = float(line[1])
                    else:
                        conf.append((line[0], line[1]))
            used_settings.append((conf, val_acc))
            if val_acc > current_best[1]:
                current_best = [conf, val_acc]
                print("\n*** New best setting *** ")
                print(conf)
                print("Validation accuracy = "+str(val_acc)+"\n\n")
            
            f.close()
            conf = hpconf.get_next_setting()
            
            ## cook new conf file
            f = open(logsdir+'log'+str(COUNT), 'a'); COUNT += 1
            print("Writing new setting:")
            print(str(conf)+"\n")
            for hp in conf:
                f.write(str(hp[0])+", "+str(hp[1])+"\n")
            f.write("\n")
            f.close()
            
            ## rename so we know we've already seen it
            os.rename(logsdir+logfile, logsdir+logfile+'.seen')
            break
    
