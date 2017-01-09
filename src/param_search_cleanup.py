import os
import shutil

report_dir = '../report'
models_dir = '../models'
run_dir_pattern = 'run_'

for d in [report_dir, models_dir]:
    print 'Cleaning up {}'.format(d)    
    for f in os.listdir(d):
        if run_dir_pattern in f:
            print 'Removing {}'.format(f)
            shutil.rmtree(os.path.join(d,f));
print 'Cleanup complete'
