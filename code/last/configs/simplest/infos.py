def create2():

    for dataType in ['normal', 'noisy']:
        for dt in [0.001]:
            for lr in [0.001]:
                for batch in [32]:

                    for path_data in ['classic']:
                        for layer_norm in [0]:
                            for dropout in [0, 1]:
                                for nb_layer in [0]:
                                    for dataUpdate in ['delta']:
                                        for tr in ['1-step']:
                                            for modelName in ['simplest']:
                                                for scaleL1 in [0.01, 0.001, 0.0001, 0.00001]:
                                        
                                                    
                                                    wb_name = f'mt-{modelName}_{dataType}_scaleL1-{scaleL1}_dropout-{dropout}'

                                                    cfg = Config()
                                                    cfg.training.batch = batch
                                                    cfg.training.lr = lr
                                                    cfg.training.wbName = wb_name
                                                    cfg.training.training_type = tr

                                                    cfg.training.loss.l1Reg = scaleL1

                                                    

                                                    cfg.training.tag.append(modelName)
                                                    cfg.training.tag.append(f'layer_norm-{layer_norm}')
                                                    cfg.training.tag.append(f'dropout-{dropout}')
                                                    cfg.training.tag.append(f'nb_layer-{nb_layer}')

                                                    

                                                    

                                                    #if 'relu-exp' in additional:
                                                    #    modelName = modelName + '-relu'

                                                    cfg.training.modelName = modelName
                                                    cfg.training.saveModel = modelName + '_latest.pt'
                                                    cfg.training.evalModel = modelName + '_best.pt'

                                                    cfg.training.loss.l1Reg = scaleL1

                                                    cfg.training.tag.append(dataUpdate)
                                                    cfg.training.tag.append(dataType)

                                                    if path_data in ['classic']:
                                                        if dt == 0.001:
                                                            print('oo')
                                                            if dataUpdate == 'delta':
                                                                if dataType == 'normal':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/mew_0.001_normal_v2'
                                                                elif dataType == 'noisy':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/mew_0.001_noisy_v2'
                                                                else:
                                                                    print('ISSUE NORMAL-NOISY')

                                                            elif dataUpdate == 'v':
                                                                cfg.training.dt_update = dt
                                                                if dataType == 'normal':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/aquali_speed_upd/aquali_0.001_normal_v2'
                                                                elif dataType == 'noisy':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/aquali_speed_upd/aquali_0.001_noisy_v2'
                                                                else:
                                                                    print('ISSUE NORMAL-NOISY')

                                                        elif dt == 0.01:
                                                            if dataUpdate == 'delta':
                                                                if dataType == 'normal':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/mew_0.01_normal_v2'
                                                                elif dataType == 'noisy':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/mew_0.01_noisy_v2'
                                                                else:
                                                                    print('ISSUE NORMAL-NOISY')

                                                            elif dataUpdate == 'v':
                                                                cfg.training.dt_update = dt
                                                                if dataType == 'normal':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/aquali_speed_upd/aquali_0.01_normal_v2'
                                                                elif dataType == 'noisy':
                                                                    cfg.training.pathData = '/scratch/users/jpierre/aquali_speed_upd/aquali_0.01_noisy_v2'
                                                                else:
                                                                    print('ISSUE NORMAL-NOISY')

                                                    new_folder = os.path.join(OUTPUT_FILE, wb_name)
                                                    makedir(new_folder)

                                                    config_mods = give_arch()

                                                    if modelName == 'gat':

                                                        config_mods['GAT_CFG']['nb_layers'] = nb_layer


                                                        if dropout:
                                                            config_mods['GAT_CFG']['encoder']['dropout'] = 0.5
                                                            config_mods['GAT_CFG']['layer']['dropout'] = 0.5
                                                            config_mods['GAT_CFG']['decoder']['dropout'] = 0.5

                                                        if layer_norm:
                                                            config_mods['GAT_CFG']['layer_norm'] = 1


                                                        config_mods['model_name'] = 'GAT_CFG'


                                                    if modelName == 'compex':

                                                        config_mods['GNN_CFG']['nb_layers'] = nb_layer


                                                        if dropout:
                                                            config_mods['GNN_CFG']['encoder']['dropout'] = 0.5
                                                            config_mods['GNN_CFG']['layer']['dropout'] = 0.5
                                                            config_mods['GNN_CFG']['decoder']['dropout'] = 0.5

                                                        if layer_norm:
                                                            config_mods['GNN_CFG']['layer_norm'] = 1


                                                        config_mods['model_name'] = 'GNN_CFG'


                                                    if modelName == 'baseline':

                                                        

                                                        if dropout:
                                                            config_mods['BASELINE_CFG']['MLP_update']['dropout'] = 0.5
                                                            config_mods['BASELINE_CFG']['MLP_message']['dropout'] = 0.5


                                                        config_mods['model_name'] = 'BASELINE_CFG'



                                                    if modelName == 'simplest':

                                                        

                                                        if dropout:
                                                            config_mods['INTERACTION_CFG']['MLP_update']['dropout'] = 0.5
                                                            config_mods['INTERACTION_CFG']['MLP_message']['dropout'] = 0.5


                                                        config_mods['model_name'] = 'INTERACTION_CFG'



                                                    cfg.training.cfg_mod = config_mods

                                                    
                                                    createYaml(cfg.to_dict(), os.path.join(new_folder, 'cfg'))
                                                    


                                                    cfg.training.tag = []
