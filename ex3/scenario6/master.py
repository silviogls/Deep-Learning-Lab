import numpy as np
import os

## hp object returns the next_configuration as a dictionary of hyperparameters.
## it computes the new configuration based on some search/optimization algorithm (eg random search)



class HPConfiguration:
    def __init__(self):
        self.hps = [('n_filters', self.get_hp_gen(10,100,10)), 
                    ('learning_rate', self.get_hp_gen(1e-2,4e-1,1e-2)), 
                    ('momentum', self.get_hp_gen(0,1,1e-1)), 
                    ('batch_size', self.get_hp_gen(16,256,15))]    
        self.configuration = [(hp[0],hp[1].next()) for hp in self.hps]
        self.current_hp_idx = 0
        
        ## TODO: make version with list or other steps... eliminate the need for numpy?
    def get_hp_gen(self, start, stop, step):
        for i in np.arange(start, stop+step, step):
            yield(i)
        
    def get_next_configuration(self):
        ret = list(self.configuration)
        ## todo: manage end of iterator exception
        self.configuration[self.current_hp_idx] = (self.hps[self.current_hp_idx][0], self.hps[self.current_hp_idx][1].next())
        self.current_hp_idx = (self.current_hp_idx + 1) % len(self.hps)
        return ret
        

## we are going to put ONE configuration in each file (makes it simpler to code)
if not os.path.exists('logs/'):
    os.makedirs('logs/')
COUNT = 0
hpconf = HPConfiguration()
while(COUNT < 2):
    f = open('logs/log'+str(COUNT), 'a'); COUNT += 1
    conf = hpconf.get_next_configuration()
    for hp in conf:
        f.write(str(hp[0])+", "+str(hp[1])+"\n")
    f.write("\n\n")
    f.close()
    
    

    
