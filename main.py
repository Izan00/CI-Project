import numpy as np
import time
from ga import generate_ga
from pygad import pygad
import yaml
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
from car_sim.config_variables import config_file_name
from matplotlib import pyplot as plt
from analysis import single_rep_analysis_plot,multi_rep_analysis_plot

np.set_printoptions(linewidth=np.inf)
simplefilter("ignore", category=ConvergenceWarning)

if __name__ == '__main__':
    # Config file reading
    with open('configs/' + config_file_name, 'r') as f:
        ga_config = yaml.safe_load(f)

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
            ga_instances.append(ga_instance)
        multi_rep_analysis_plot(ga_instances, ga_config)
        plt.savefig(ga_config['load_file_name'] + 'multi.png')
        del ga_instances
    plt.show()
