import numpy as np
from matplotlib import pyplot as plt


def exrtract_fitness_data(ga_instance):
    # ga_instance.plot_fitness()
    solutions = np.array(ga_instance.solutions)
    solutions = np.reshape(solutions, (ga_instance.num_generations, -1, solutions.shape[1]))
    solutions_fitness = ga_instance.solutions_fitness
    best_solution = ga_instance.best_solutions

    solution_variance = []
    for gen in range(solutions.shape[0]):
        solutions_gen = solutions[gen][:][:]
        solution_gen_variance = []
        for i in range(solutions_gen.shape[0]):
            for j in range(i + 1, solutions_gen.shape[0]):
                solution_gen_variance.append(np.linalg.norm(solutions_gen[i] - solutions_gen[j]))
        solution_variance.append(np.mean(solution_gen_variance))

    best_solution_fitness = ga_instance.best_solutions_fitness
    solutions_fitness = np.reshape(solutions_fitness, (ga_instance.num_generations, -1))
    solutions_fitness_mean = np.mean(solutions_fitness, axis=1)

    return solutions_fitness, solutions_fitness_mean, best_solution_fitness, solution_variance

def single_rep_analysis_plot(ga_instance):

    solutions_fitness, solutions_fitness_mean, best_solution_fitness, solution_variance = exrtract_fitness_data(ga_instance)

    x = [i for i in range(1, ga_instance.num_generations+1)]
    x_ticks = [i for i in range(1,ga_instance.num_generations+1, 5)]
    if ga_instance.num_generations > 100:
        fig, ax = plt.subplots(3, 1, figsize=(18, 6), constrained_layout=True)
    else:
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), constrained_layout=True)
    # fig.suptitle('a')
    ax[0].scatter(x, solutions_fitness_mean, marker='+', c='#1f77b4', s=60)
    ax[0].scatter(x, best_solution_fitness, marker='1', c='#ff7f0e', s=60)
    ax[0].set_title('Best fitness')
    ax[0].set_ylabel('Fitness')
    ax[0].set_xlabel('Generation')
    ax[0].legend(['Mean', 'Best'], loc='center right')
    ax[0].grid(alpha=0.7, linestyle='-')
    ax[0].set_xticks(x_ticks)

    for i in range(solutions_fitness.shape[1]):
        ax[1].scatter(x, solutions_fitness.T[i], marker='+', c='#1f77b4', s=60)
    ax[1].set_title('Fitness vs Generation')
    ax[1].set_ylabel('Fitness')
    ax[1].set_xlabel('Generation')
    ax[1].grid(alpha=0.7, linestyle='-')
    ax[1].set_xticks(x_ticks)

    ax[2].scatter(x, solution_variance, marker='+', c='#1f77b4', s=60)
    ax[2].set_title('Average distance between individuals')
    ax[2].set_ylabel('Average distance')
    ax[2].set_xlabel('Generation')
    ax[2].grid(alpha=0.7, linestyle='-')
    ax[2].set_xticks(x_ticks)

def multi_rep_analysis_plot(ga_instances, ga_config):
    highest_fitness_achievement = []
    highest_fitness_achievement_null = []
    for instance in ga_instances:
        max_fitness_reached = np.max(np.array(instance.best_solutions_fitness) >= ga_config['max_fitness_kill'])
        max_fitness_index = np.argmax(np.array(instance.best_solutions_fitness) >= ga_config['max_fitness_kill'])
        if max_fitness_reached:
            highest_fitness_achievement.append(max_fitness_index)
            highest_fitness_achievement_null.append(np.nan)
        else:
            highest_fitness_achievement_null.append(0)
            highest_fitness_achievement.append(np.nan)

    #highest_fitness_achievement[-1]=np.nan
    #highest_fitness_achievement_null[-1]=-1

    x = [i for i in range(1,len(highest_fitness_achievement)+1)]
    x_ticks = [i for i in range(1, len(highest_fitness_achievement)+1, 2)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    ax.scatter(x, highest_fitness_achievement, marker='+', c='#1f77b4', s=80)
    ax.scatter(x, highest_fitness_achievement_null, marker='+', c='#ff7f0e', s=80)
    ax.set_title('Max fitness achievement')
    ax.set_ylabel('Generation')
    ax.set_xlabel('Experiment ID')
    ax.set_xticks(x_ticks)
    ax.grid(alpha=0.7, linestyle='-')
    ax.legend(['Max reached', 'Max NOT reached'], loc='upper left')