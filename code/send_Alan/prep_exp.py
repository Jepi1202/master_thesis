import zipfile
import shutil
import os
import sys
import yaml


BASE_FILE = '/master/code/send_Alan/cfg_gen'                    # input file of the cfgs
BASH_FOLDER = '/master/code/send_Alan/bash_script'              # output path of the bash scripts
FUN_FOLDER = '/master/code/send_Alan/base_functions'            # input path of the functions
OUT_FOLDER = '/master/code/send_Alan/out'                       # output file for everything to be sent

#NB_PER_ALAN = 5
NB_PER_ALAN = 16



def create_bash(path_sh, path_data, Alan_nb):
    bash_script_content = f"""#!/bin/bash

    cd /home/jpierre/v2/new_runs/Alan/{Alan_nb}/{path_data}
    eval "$(conda shell.bash hook)"
    conda activate myenvPy
    python training_main.py
    """

    with open(path_sh, 'w') as file:
        file.write(bash_script_content)


    st = os.stat(path_sh)
    os.chmod(path_sh, st.st_mode | 0o755)



def create_zip(zip_filename, files):
    # check if works with folders on top of files
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file)




def createEnvs(path_base, nb_Alan, path_bash, path_functions, path_out):
    
    nb = 0      # index of the number of files in an Alan
    i = 0       # nb of total files
    
    # get all the cfgs
    folders = os.listdir(path_base)

    # get all the functions
    function_files = os.listdir(path_functions)

    # initialize the Alan folder
    p_alan = os.path.join(path_out, 'Alan')
    if not os.path.exists(p_alan):
        os.makedirs(p_alan)

    # initialize the bash folders
    path_bash = os.path.join(path_bash, 'Alan_bash')
    if not os.path.exists(path_bash):
        os.makedirs(path_bash)

    #for p_b in ['Alan1', 'Alan2', 'Alan3', 'Alan4']:
    for p_b in ['Alan3']:
    #for p_b in ['Alan1', 'Alan4']:

        # create the folder for the Alan i (in bash and in the run place)
        p_base = os.path.join(p_alan, p_b)
        p_base_bash = os.path.join(path_bash, p_b)

        # create Alan_{i}
        if not os.path.exists(p_base):
            os.makedirs(p_base)

        if not os.path.exists(p_base_bash):
            os.makedirs(p_base_bash)

        while(i < len(folders)):
            if nb == nb_Alan:
                nb = 0
                break

            # path of the config
            p_cfg = os.path.join(path_base, folders[i])
            p_cfg = os.path.join(p_cfg, 'cfg.yml')

            # path of the folder with everything (output)
            p = os.path.join(p_base, folders[i])
    
            if os.path.exists(p):
                print('ISSUE HERE')

            os.makedirs(p)
            p_cfg2 = os.path.join(p, 'cfg.yml')
            shutil.move(p_cfg, p_cfg2)
            shutil.rmtree(os.path.join(path_base, folders[i]))      # get rid of the stuff that are created before

            os.makedirs(os.path.join(p, 'model_trained'))   # create folder for the models

            # copy the functions
            for fun in function_files:
                p_fun = os.path.join(path_functions,fun)
                p_fun2 = os.path.join(p,fun)

                shutil.copy(p_fun, p_fun2)


            # create the corresponding bash files
            p_bash = os.path.join(p_base_bash, f'exp_{i}.sh')
            create_bash(p_bash, folders[i], p_b)

            i += 1
            nb += 1

        # zip the file for the given Alan -- manually do that for now ...

        #create_zip


def main():
    createEnvs(BASE_FILE, NB_PER_ALAN, BASH_FOLDER, FUN_FOLDER, OUT_FOLDER)

if __name__ == '__main__':
    main()