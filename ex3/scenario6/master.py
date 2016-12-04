import numpy as np
import os
import csv
from datetime import datetime
## hp object returns the next_configuration as a dictionary of hyperparameters.
## it computes the new configuration based on some search/optimization algorithm (eg random search)



class HPConfiguration:
    def __init__(self):
        self.hps = [('nfilters', self.get_hp_gen(10,100,10)), 
                    ('LR', self.get_hp_gen(1e-2,4e-1,1e-2)), 
                    ('M', self.get_hp_gen(0,1,1e-1)), 
                    #('batch_size', self.get_hp_gen(16,256,15))
                    ]    
        self.configuration = [(hp[0],hp[1].next()) for hp in self.hps]
        self.current_hp_idx = 0
        
        ## TODO: make version with list or other steps... eliminate the need for numpy?
        
    def get_hp_gen(self, start, stop, step):
        for i in np.arange(start, stop+step, step):
            yield(i)
        
        ## MAKE RANDOM SEARCH !!!
        
        ## TODO: be aware of when the combinations are over (ok but with random search no more problems...)
    def get_next_configuration(self):
        ret = list(self.configuration)
        try:
            newval = self.hps[self.current_hp_idx][1].next()
        except StopIteration:
            newval = self.configuration[self.current_hp_idx][1]
            
        self.configuration[self.current_hp_idx] = (self.hps[self.current_hp_idx][0], newval)
        self.current_hp_idx = (self.current_hp_idx + 1) % len(self.hps)
        return ret
        

## we are going to put ONE configuration in each file (makes it simpler to code)
logsdir = "logs"+str(datetime.today()).replace(" ","_").replace(":","_").replace(".","_")+"/"
if not os.path.exists(logsdir):
    os.makedirs(logsdir)
COUNT = 0
N_WORKERS = 2
hpconf = HPConfiguration()

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

while(True):
    ## read returned performances and make new config files
    for logfile in os.listdir(logsdir):
        if 'log' in logfile and '.done' in logfile and '.seen' not in logfile:
            ## read performance
            f = open(logsdir+logfile, "r")
            reader = csv.reader(f)
            for line in reader:
                if len(line) == 2 and 'best validation accuracy' in line[0] and float(line[1]) > current_best[1]:
                    current_best[1] = float(line[1])
            f.close()
            print(current_best)
            
            ## cook new conf file
            f = open(logsdir+'log'+str(COUNT), 'a'); COUNT += 1
            conf = hpconf.get_next_configuration()
            print(conf)
            for hp in conf:
                f.write(str(hp[0])+", "+str(hp[1])+"\n")
            f.write("\n")
            f.close()
            
            ## rename so we know we've already seen it
            os.rename(logsdir+logfile, logsdir+logfile+'.seen')
            break
    
