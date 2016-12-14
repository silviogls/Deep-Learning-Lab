import csv
import os
import exercise3

## just cycling through filenames doesn't make sense because the files get regularly updated! TODO: change this
while(True):
    for log in os.listdir('logs'):
        if 'log' in log and 'taken' not in log:
            print("\nrunning "+log)
            os.rename('logs/'+log, 'logs/'+log+'.taken')
            f = open('logs/'+log+'.taken', 'r')
            reader = csv.reader(f)
            setting = dict()
            for hp in reader:
                if len(hp) == 2:
                    val = float(hp[1])
                    if 'batch_size' in hp[0] or 'filters' in hp[0] or 'num_epochs' in hp[0]:
                        val = int(val)
                    setting[hp[0]] = val
                else: break
            print(setting)
            # run setting
            conf_run = exercise3.main(**setting)    ## num_epochs = 2 ofc just for now...
            # append performance to file...
            
            f.close()
            f = open('logs/'+log+'.taken', 'a')
            f.write('best validation accuracy, '+str(conf_run[0])+'\n')
            f.close()
            os.rename('logs/'+log+'.taken', 'logs/'+log+'.taken'+'.done')
            break   ## need to break the cycle because the files have been updated
