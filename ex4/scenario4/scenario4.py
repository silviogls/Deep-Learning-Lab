import os
import time
import csv
import math
import numpy as np
import re

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


    
def run_model_distributed(hyperparameters, nepochs):
    ## manage logs directory
    logsdir = "logs/"
    if not os.path.exists(logsdir):
        os.makedirs(logsdir)
    for l in os.listdir(logsdir):
        os.remove(logsdir+l)

    print("Writing " +str(len(hyperparameters))+ " settings to be evaluated:")
    ## write configurations in files (all of them)
    for (index, hps) in enumerate(hyperparameters):
        setting = list(hps)
        setting.append(['num_epochs', int(nepochs)])
        print("log"+str(index)+":   "+str(setting))
        ## cook new conf file
        f = open(logsdir+'log'+str(index), 'a')
        for hp in setting:
            f.write(str(hp[0])+", "+str(hp[1])+"\n")
        f.write("\n")
        f.close()
        
    validations = []
    while len(validations) < len(hyperparameters):
        time.sleep(0.05) ## cpu rest
        
        ## read returned performances and make new config files
        for logfile in os.listdir(logsdir):
            if 'log' in logfile and '.done' in logfile and '.seen' not in logfile:
                ## read performance
                
                index = int(re.findall('\d+', logfile)[0]) ## extract index from filename
                f = open(logsdir+logfile, "r")
                reader = csv.reader(f)

                for line in reader:
                    if len(line) == 2 and 'best validation accuracy' in line[0]:
                            val_acc = float(line[1])
                validations.insert(index, val_acc)
                f.close()
                ## rename so we know we've already seen it
                os.rename(logsdir+logfile, logsdir+logfile+'.seen')
                break
    print("DONE")
    return validations

### Hyperband routine

def hyperband(get_random_hyperparameter_configuration, run_then_return_val_loss, max_iter=20, eta=3):
    #~ max_iter = 81   # maximum iterations/epochs per configuration
    #~ eta = 3         # defines downsampling rate (default=3)
    logeta = lambda x: math.log(x)/math.log(eta)
    s_max = int(logeta(max_iter))   # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter          # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    noiselevel = 0.2  # noise level of the objective function
    # Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.

    nruns = 1       # set it to e.g. 10 when testing hyperband against randomsearch
    for irun in range(0, nruns):
        if not os.path.exists('hb'):
            os.makedirs('hb')
        hband_results_filename = "hb/hyperband_{}.txt".format(irun)
        hband_file = open(hband_results_filename, 'w+', 0)

        x_best_observed = []
        x_best_observed_nep = 0

        nevals = 0       # total number of full (with max_iter epochs) evaluations used so far

        for s in reversed(range(s_max+1)):

            stat_filename = "hb/hband_benchmark_{}.txt".format(s)
            stat_file = open(stat_filename, 'w+', 0)

            n = int(math.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
            print("Initial number of configurations:  "+str(n))
            r = max_iter*eta**(-s)      # initial number of iterations to run configurations for

            # Begin Finite Horizon Successive Halving with (n,r)
            T = [ get_random_hyperparameter_configuration() for i in range(n) ]
            
            for i in range(s+1):
                
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = r*eta**(i)
                #print("###  "+str(i)+" of "+str(s+1)+":  "+str(r_i))
                #print("n of configurations:  "+str(len(T)))
                val_losses = run_then_return_val_loss(T, r_i)
                nevals = nevals + len(T) * r_i / max_iter
                argsortidx = np.argsort(val_losses)

                if (x_best_observed == []):
                    x_best_observed = T[argsortidx[0]]
                    y_best_observed = val_losses[argsortidx[0]]
                    x_best_observed_nep = r_i
                # only if better AND based on >= number of epochs, the latter is optional
                if (val_losses[argsortidx[0]] < y_best_observed):# and (r_i >= x_best_observed_nep):
                    x_best_observed_nep = r_i
                    y_best_observed = val_losses[argsortidx[0]]
                    x_best_observed = T[argsortidx[0]]

                for j in range(0, len(T)):
                    #stat_file.write("{:.15g}\t{:.15g}\n".format(T[j], val_losses[j]))
                    stat_file.write(str(T[j])+"\t"+str(val_losses[j]))
                T = [ T[i] for i in argsortidx[0:int( n_i/eta )] ]
                
                if len(T) == 1:
                    print("Chosen setting:   "+str(T[0]))
                
                # suppose the current best solution w.r.t. validation loss is our recommendation
                # then let's evaluate it in noiseless settings (~= averaging over tons of runs)
                #~ if (len(T)):
                    #~ print("########### FINAL")
                    #~ f_recommendation = run_then_return_val_loss(max_iter, x_best_observed, 1e-10) # full epochs and 1e-10 ~= zero noise
                #~ hband_file.write("{:.15g}\t{:.15g}\n".format(nevals, f_recommendation))
            print("\n")
            # End Finite Horizon Successive Halving with (n,r)


            stat_file.close()
        hband_file.close()

### MAIN

hpconf = HPConfiguration([
                    ('nfilters', 10, 100, 10), 
                    ('LR', 0.01, 0.4, 0.02), 
                    ('M', 0, 1, 1e-1), 
                    ('batch_size_train', 32, 256, 32),
                    ('ntrain', 501,502,1)
                    ], mode='random')

hyperband(hpconf.get_next_setting, run_model_distributed)
