
import yaml

# Load config file
def Getyaml(filename='config.yml'):
    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_dict={'Layer':[]}

    # Access hyperparameters
    config_dict['learning_rate'] = config['learning_rate']
    config_dict['batch_size'] = config['batch_size']
    config_dict['num_epochs'] = config['num_epochs']
    config_dict['dataset_path'] = config['dataset']['path']
    config_dict['train_split'] = config['dataset']['train_split']
    config_dict['val_split'] = config['dataset']['val_split']
    config_dict['Loss_function'] = config['Loss_function']
    config_dict['momentom'] = config['momentom']
    config_dict['Normalize'] = config['Normalize']
    i=1
    while(True):
        config_dict['Layer'].append([config[f'Layer_{i}'].get('Activation_function','ReLu'),config[f'Layer_{i}'].get('num_classes',-1),config[f'Layer_{i}'].get('mu',0),config[f'Layer_{i}'].get('sigma',0.01),config[f'Layer_{i}'].get('bias',0)])
        if(config[f'Layer_{i}'].get('num_classes',-1)==-1):
            break
        i+=1
    config_dict['Layer'][-1][1]=config['dataset']['num_classes']

    return config_dict