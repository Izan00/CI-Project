import numpy as np
import sys
import yaml
from car_sim.config_variables import config_load
from matplotlib import pyplot as plt
from analysis import single_rep_analysis_plot,multi_rep_analysis_plot
from pygad import pygad

#np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=False)

if __name__ == '__main__':

    #default config file name
    config_file_name = 'default_config.yaml'

    # Arguments reading
    args = sys.argv
    for flag,arg in zip(args,args[1:]):
        if flag == '-c':
            config_file_name=arg

    # Config file reading
    with open('configs/' + config_file_name, 'r') as f:
        ga_config = yaml.safe_load(f)

    # update config and load the genetic experiment library
    config_load(ga_config)
    from ga import generate_ga

    if ga_config['load_file_name'] == 'None':
        ga_instance = generate_ga(ga_config)
        ga_instance.run()
        ga_instance.plot_fitness()
        if not ga_config['simulation_save_file'] == 'None':
            save_file_name = ga_config['simulation_save_file']
            ga_instance.save(filename=save_file_name)
        single_rep_analysis_plot(ga_instance)

    elif ga_config['multi_file'] == 0:
        load_file_name = ga_config['load_file_name']
        ga_instance = pygad.load(filename=load_file_name)
        single_rep_analysis_plot(ga_instance)
        plt.savefig(ga_config['load_file_name'] + '.png')

    else:
        ga_instances = []
        for i in range(1,ga_config['multi_file']+1):
            load_file_name = ga_config['load_file_name']+str(i)
            ga_instance = pygad.load(filename=load_file_name)
            #single_rep_analysis_plot(ga_instance)
            #plt.savefig(load_file_name + '.png')
            #plt.close('all')
            ga_instances.append(ga_instance)
        multi_rep_analysis_plot(ga_instances, ga_config)
        plt.savefig(ga_config['load_file_name'] + 'multi.png')
        del ga_instances
    plt.show()
