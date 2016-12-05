import numpy as np
import os
import csv
from datetime import datetime
import time
## hp object returns the next_configuration as a dictionary of hyperparameters.
## it computes the new configuration based on some search/optimization algorithm (eg random search)



class HPConfiguration:
    def __init__(self, hps_init, mode = 'random'): 
        self.hps_init = hps_init
        self.hps = []
        
        if mode == 'random':
            self.get_next_configuration = self.get_next_configuration_random
        elif mode == 'random_search':
            self.get_next_configuration = self.get_next_configuration_random_search
        elif mode == 'grid':
            for hp in hps_init:
                self.hps.append((hp[0], self.get_hp_gen(hp[1],hp[2],hp[3])))
            self.get_next_configuration = self.get_next_configuration_grid
            
        self.n_hps = len(self.hps_init)
        self.configuration = [(hp[0],hp[1].next()) for hp in self.hps]
        self.results = []
        self.current_hp_idx = 0
        self.current_best = [None, 0]
        self.has_more = 0
        ## TODO: make version with list or other steps... eliminate the need for numpy?
        
    def get_hp_gen(self, start, stop, step):
        for i in np.arange(start, stop+step, step):
            yield(i)
        
    def has_more_configurations(self):
        return self.has_more < self.n_hps
        
    def get_next_configuration_random(self):
        ret = []
        for hp in self.hps_init:
            ret.append([hp[0], hp[1]+np.random.rand()*(hp[2]-hp[1])])
        return ret

    def get_next_configuration_grid(self):
        ret = list(self.configuration)
        try:
            newval = self.hps[self.current_hp_idx][1].next()
        except StopIteration:
            self.has_more += 1
            newval = self.configuration[self.current_hp_idx][1]
            
        self.configuration[self.current_hp_idx] = (self.hps[self.current_hp_idx][0], newval)
        self.current_hp_idx = (self.current_hp_idx + 1) % len(self.hps)
        return ret
        
    ## random search, based on best result so far
    def get_next_configuration_random_search(self, new_performance, config):
        ret = None
        raise NotImplementedError
        ## save in results
        ## compare with best
        ## return new configuration
        return ret
        

## we are going to put ONE configuration in each file (makes it simpler to code)
#~ logsdir = "logs"+str(datetime.today()).replace(" ","_").replace(":","_").replace(".","_")+"/"
logsdir = "logs"+"/"
if not os.path.exists(logsdir):
    os.makedirs(logsdir)
COUNT = 0
N_WORKERS = 2
hpconf = HPConfiguration([
                    ('nfilters', 10, 100, 10), 
                    ('LR', 0.01, 0.4, 0.02), 
                    ('M', 0, 1, 1e-1), 
                    ('batch_size_train', 32, 256, 32)
                    ])

## make initial configuration files
for i in range(N_WORKERS):
    f = open(logsdir+'log'+str(COUNT), 'a'); COUNT += 1
    conf = hpconf.get_next_configuration()
    print(conf)
    for hp in conf:
        f.write(str(hp[0])+", "+str(hp[1])+"\n")
    f.write("\n")
    f.close()

current_best = [None, 0]

while(hpconf.has_more_configurations()):
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
                    
            f.close()
            conf = hpconf.get_next_configuration()
            
            ## cook new conf file
            f = open(logsdir+'log'+str(COUNT), 'a'); COUNT += 1
            print(conf)
            for hp in conf:
                f.write(str(hp[0])+", "+str(hp[1])+"\n")
            f.write("\n")
            f.close()
            
            ## rename so we know we've already seen it
            os.rename(logsdir+logfile, logsdir+logfile+'.seen')
            break
    
