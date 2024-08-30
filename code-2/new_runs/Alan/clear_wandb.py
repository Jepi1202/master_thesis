import os
import shutil


#PATH = '/home/jpierre/v2/new_runs/Alan/previous_exp'
PATH = [#'/home/jpierre/v2/new_runs/Alan/Alan2',
        #'/home/jpierre/v2/new_runs/Alan/Alan1',
        #'/home/jpierre/v2/new_runs/Alan/Alan3',
        #'/home/jpierre/v2/new_runs/Alan/Alan4',
        '/home/jpierre/v2/new_runs/Alan/previous_exp'
       ]

def delete_wandb_dirs(start_path):
    for root, dirs, files in os.walk(start_path, topdown=False):
        for dir_name in dirs:
            if dir_name == "wandb":
                dir_path = os.path.join(root, dir_name)
                print(f"Deleting: {dir_path}")
                shutil.rmtree(dir_path)

                
if __name__ == '__main__':

    for p in PATH:
        delete_wandb_dirs(p)