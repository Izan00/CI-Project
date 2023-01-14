import numpy as np
import time
from ga import generate_ga
from pygad import pygad
import yaml
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
from car_sim.config_variables import config_file_name
from matplotlib import pyplot as plt

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

    else:
        load_file_name = ga_config['load_file_name']
        ga_instance = pygad.load(filename=load_file_name)

    #ga_instance.plot_fitness()
    solutions = np.array(ga_instance.solutions)
    solutions = np.reshape(solutions, (ga_instance.num_generations, -1, solutions.shape[1]))
    solutions_fitness = ga_instance.solutions_fitness
    best_solution = ga_instance.best_solutions

    solution_variance = []
    for gen in range(solutions.shape[0]):
        solutions_gen=solutions[gen][:][:]
        solution_gen_variance=[]
        for i in range(solutions_gen.shape[0]):
            for j in range(i + 1, solutions_gen.shape[0]):
                solution_gen_variance.append(np.linalg.norm(solutions_gen[i] - solutions_gen[j]))
        solution_variance.append(np.mean(solution_gen_variance))

    best_solution_fitness = ga_instance.best_solutions_fitness
    solutions_fitness=np.reshape(solutions_fitness, (ga_instance.num_generations,-1))
    solutions_fitness_mean=np.mean(solutions_fitness, axis=1)

    x = [i + 1 for i in range(ga_instance.num_generations)]
    print(solutions_fitness.shape)
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), constrained_layout=True)
    # fig.suptitle('a')
    ax[0].scatter(x, solutions_fitness_mean, marker='+', c='#1f77b4', s=60)
    ax[0].scatter(x,best_solution_fitness,marker='x', c='#ff7f0e',s=60)
    ax[0].set_title('Best fitness')
    ax[0].set_ylabel('Fitness')
    ax[0].set_xlabel('Generation')
    ax[0].legend(['Mean','Best'],loc='upper left')
    for i in range(solutions_fitness.shape[1]):
        ax[1].scatter(x,solutions_fitness.T[i],marker='+', c='#1f77b4',s=60)
    ax[1].set_title('Fitness vs Generation')
    ax[1].set_ylabel('Fitness')
    ax[1].set_xlabel('Generation')

    ax[2].scatter(x, solution_variance, marker='+', c='#1f77b4', s=60)
    ax[2].set_title('Average distance between individuals')
    ax[2].set_ylabel('Average distance')
    ax[2].set_xlabel('Generation')

    plt.savefig(config_file_name+'.png')

    plt.show()
