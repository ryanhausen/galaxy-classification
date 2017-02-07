import os
import shutil

report_dir = '../report'
models_dir = '../models'
run_dir_pattern = 'run_'

print 'Cleaning up log and best params'
for f in ['param_search_log', 'best_params.json']:
    if f in os.listdir('.'):
        print 'Removing {}'.format(f)
        os.remove(f)

for d in [report_dir, models_dir]:
    print 'Cleaning up {}'.format(d)    
    for f in os.listdir(d):
        if run_dir_pattern in f:
            print 'Removing {}'.format(f)
            shutil.rmtree(os.path.join(d,f))
print 'Cleanup complete'
