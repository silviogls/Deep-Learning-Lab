import exercise3
import math
import numpy as np

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
        ret = {}
        for hp in self.hps_init:
            val = hp[1]+np.random.rand()*(hp[2]-hp[1])
            ## convert to int the hyperparameters when needed (batch size and number of filters)
            if 'batch_size' in hp[0] or 'filters' in hp[0]:
                val = int(val)
            ret[hp[0]] = val
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


def run_model(nepochs=10, hyperparameters=None, noiselevel=0):   ## noiselevel is kept just for compatibility with the hyperband routine...
    print("Hyperparameters:")
    print(hyperparameters)
    print("Number of epochs:  "+str(nepochs))
    ret = exercise3.main(num_epochs = int(nepochs), **hyperparameters)
    return ret[0]


### Hyperband routine

def hyperband(get_random_hyperparameter_configuration, run_then_return_val_loss, max_iter=10, eta=5):
    #max_iter = 81   # maximum iterations/epochs per configuration
    #eta = 3         # defines downsampling rate (default=3)
    logeta = lambda x: math.log(x)/math.log(eta)
    s_max = int(logeta(max_iter))   # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter          # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    noiselevel = 0.2  # noise level of the objective function
    # Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.

    nruns = 1       # set it to e.g. 10 when testing hyperband against randomsearch
    for irun in range(0, 100):
        hband_results_filename = "hyperband_{}.txt".format(irun)
        hband_file = open(hband_results_filename, 'w+', 0)

        x_best_observed = []
        x_best_observed_nep = 0

        nevals = 0       # total number of full (with max_iter epochs) evaluations used so far

        for s in reversed(range(s_max+1)):

            stat_filename = "hband_benchmark_{}.txt".format(s)
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
                
                val_losses = [ run_then_return_val_loss(nepochs=r_i, hyperparameters=t,
                                                        noiselevel=noiselevel) for t in T ]
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

                # suppose the current best solution w.r.t. validation loss is our recommendation
                # then let's evaluate it in noiseless settings (~= averaging over tons of runs)
                if (len(T)):
                    f_recommendation = run_then_return_val_loss(max_iter, x_best_observed, 1e-10) # 81 epochs and 1e-10 ~= zero noise
                hband_file.write("{:.15g}\t{:.15g}\n".format(nevals, f_recommendation))
            # End Finite Horizon Successive Halving with (n,r)

            stat_file.close()
        hband_file.close()

### MAIN

hpconf = HPConfiguration([
                    ('nfilters', 10, 100, 10), 
                    ('LR', 0.01, 0.4, 0.02), 
                    ('M', 0, 1, 1e-1), 
                    ('batch_size_train', 32, 256, 32)
                    ], mode='random')

hyperband(hpconf.get_next_setting, run_model)
