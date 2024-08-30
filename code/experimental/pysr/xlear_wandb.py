import os
import shutil


#PATH = '/home/jpierre/v2/new_runs/Alan/previous_exp'
PATH = '/master/code/experimental/pysr'

def delete_wandb_dirs(start_path):
    for root, dirs, files in os.walk(start_path, topdown=False):
        for dir_name in dirs:
            if dir_name == "wandb":
                dir_path = os.path.join(root, dir_name)
                print(f"Deleting: {dir_path}")
                shutil.rmtree(dir_path)

                
if __name__ == '__main__':

    delete_wandb_dirs(PATH)