import matplotlib.pyplot as plt
import numpy as np
import time
import math
import sklearn
from pygad import pygad
from ga import generate_ga
import yaml
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

np.set_printoptions(linewidth=np.inf)
simplefilter("ignore", category=ConvergenceWarning)


if __name__ == '__main__':
    config_file_name = 'ga_vo.yaml'
    # Config file reading
    with open('configs/' + config_file_name, 'r') as f:
        ga_config = yaml.safe_load(f)

    #load_file_name = "ga_v0"
    #ga_instance = pygad.load(filename=load_file_name)

    ga_instance = generate_ga(ga_config)

    start = time.time()

    ga_instance.run()

    exec_time = time.time() - start

    solution_values = ga_instance.best_solutions[-1]
    solution=solution_values[0]
    solution_fitness=solution_values[1]
    solution_idx=solution_values[2]

    #ga_mlp=generate_network(solution);

    print('\n')
    print('Execution time: ' + str(round(exec_time,2))+'s \n')
    if ga_instance.best_solution_generation != -1:
        print('Best fitness value reached after ' + str(ga_instance.best_solution_generation) + ' generations \n')

    ga_instance.plot_fitness()


    #save_file_name = "ga_v0"
    #ga_instance.save(filename=save_file_name)