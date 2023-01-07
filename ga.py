import numpy as np
import pygad
from sklearn.neural_network import MLPRegressor as MLP
import math
import time
from IPython.display import clear_output

import pygame as py
import os
import random
from car import Car
from road import Road
from world import World
from config_variables import *

py.font.init()
bg = py.Surface((WIN_WIDTH, WIN_HEIGHT))
bg.fill(GRAY)

'''
----------------------------
 NN_nk | ... | NN_n0 | w00 | w01 | ...
----------------------------
'''

def neurons_number_calc( instance):
    global max_neurons, max_weight_value, min_weight_value, genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop, num_parents_mating, parent_selection_type, keep_parents, k_tournament, keep_elitism
    global crossover_type, crossover_probability, mutation_type, mutation_probability, mutation_by_replacement
    global allow_duplicate_genes, verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value, min_weight_value, genetic_model, input_size, output_size, genetic_model_precision

    n = instance[:nn_n_gen].astype('int8')  # extract neurons number binary gens
    return n.dot(2 ** np.arange(n.size)[::-1]) + 1  # convert neurons number binary gens to decimal
    # TODO: change to gray encoding

def weights_number_calc( genes):  # genetic mode (binary -> decimal)
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    weights_temp = []
    for i in range(0, len(genes), w_n_gen):
        w = genes[i + 1:i + w_n_gen]
        w = np.sign(0.5 - genes[i]) * w.dot(2 ** np.arange(w.size)[::-1]) * genetic_model_precision
        weights_temp.append(w)
    return weights_temp

def generate_network( genes):
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    neurons = neurons_number_calc(genes)
    dummy_in = [0 for i in range(input_size)]
    dummy_out = [0 for i in range(output_size)]
    ga_mlp = MLP(hidden_layer_sizes=[neurons])  # Generate the network model
    ga_mlp.partial_fit(np.array(dummy_in).reshape(1, -1), np.array(dummy_out).reshape(1, -1))  # Start architecture

    ''' MLP weights/bias distribution 
    (input, nn_n) -> in-hl w
    (nn_n,)       -> hl b
    (nn_n, out)   -> hl-out w
    (out,)        -> out b
    '''

    # Parse weights
    if genetic_model:
        g = weights_number_calc(genes[nn_n_gen:nn_n_gen + input_size * neurons * w_n_gen])
        ga_mlp.coefs_[0] = np.array(g).reshape(-1, neurons)
        g = weights_number_calc(
            genes[nn_n_gen + input_size * max_neurons * w_n_gen:
                  nn_n_gen + (input_size * max_neurons + neurons) * w_n_gen])
        ga_mlp.intercepts_[0] = np.array(g)
        g = weights_number_calc(
            genes[nn_n_gen + (input_size + 1) * max_neurons * w_n_gen:
                  nn_n_gen + ((input_size + 1) * max_neurons + output_size*neurons) * w_n_gen])
        ga_mlp.coefs_[1] = np.array(g).reshape(neurons, -1)
        g = weights_number_calc(genes[-w_n_gen:])
        ga_mlp.intercepts_[1] = np.array(g)

    else:
        ga_mlp.coefs_[0] = np.array(genes[nn_n_gen:nn_n_gen + input_size * neurons]).reshape(-1, neurons)
        ga_mlp.intercepts_[0] = np.array(genes[nn_n_gen + input_size * max_neurons:nn_n_gen +
                                               input_size * max_neurons + neurons])
        ga_mlp.coefs_[1] = np.array(genes[nn_n_gen + (input_size+1) * max_neurons:
                                          nn_n_gen + (input_size+1) * max_neurons + output_size*neurons]).reshape(neurons, -1)
        ga_mlp.intercepts_[1] = np.array(genes[-output_size:])

    return ga_mlp
'''
def on_cal_pop_fitness(population):
    pop_fitness = ["undefined"] * len(population)
    for sol_idx, sol in enumerate(population):
        fitness = fitness_func(sol, sol_idx)
        pop_fitness[sol_idx] = fitness
    return pop_fitness
'''

def on_cal_pop_fitness(population):
    return car_sim(population)

def fitness_func(solution, solution_idx):
    return 0

'''
def fitness_func(solution, solution_idx):
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    ga_mlp = generate_network(solution)
    fitness=car_sim(ga_mlp)

    #print(fitness)
    return fitness
'''

def ga_run_summary(ga_instance):
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    print('Genetic algorithm MLP training')
    print('Number of generations:   {num_generations}'.format(num_generations=num_generations))
    print('Population size:         {pop_size}'.format(pop_size=ga_instance.pop_size[0]))
    print('Number of genes:         {genes}'.format(genes=ga_instance.pop_size[1]))
    print('Mutation type:           {mutation_type}'.format(mutation_type=ga_instance.mutation_type))
    print('Mutation probability:    {mutation_probability}'.format(mutation_probability=ga_instance.mutation_probability))
    print('Crossover type:          {crossover_type}'.format(crossover_type=ga_instance.crossover_type))
    print('Crossover probability:   {crossover_probability}'.format(crossover_probability=ga_instance.crossover_probability))
    print('\n')

def callback_start(ga_instance):
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    global world, road, GEN

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    world.win.blit(bg, (0, 0))
    road = Road(world)
    GEN = 1
    last_fitness = 0
    ga_run_summary(ga_instance)

def callback_generation( ga_instance):
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    global road, GEN
    GEN+=1
    road = Road(world)

    if ga_instance.generations_completed % verbose == 0:
        neurons = neurons_number_calc(ga_instance.best_solution()[0])
        #clear_output(wait=True)
        #ga_run_summary(ga_instance)
        print('Generation {generation}'.format(generation=ga_instance.generations_completed))
        print('Last gen parents:        {par_id}'.format(par_id=ga_instance.last_generation_parents_indices))
        print('Best fitness:            {fitness}'.format(fitness=round(ga_instance.best_solution()[1],2)))
        print('Best number of neurons:  {neurons}'.format(neurons=neurons))
        print('Best genes:              {genes}'.format(genes=np.around(ga_instance.best_solution()[0], decimals=2)))
        if verbose == 1:
            print('Best genes change:       {change}'.format(change=np.around(ga_instance.best_solution()[0] - last_fitness, decimals=2)))
            last_fitness = ga_instance.best_solution()[0]
        print('\n')
    time.sleep(time_delay)


def draw_win(cars, road, world):     #x e y sono le coordinate della macchina migliore
    global GEN
    road.draw(world)
    for car in cars:
        car.draw(world)

    text = STAT_FONT.render("Best Car Score: "+str(int(world.getScore())), 1, BLACK)
    world.win.blit(text, (world.win_width-text.get_width() - 10, 10))
    text = STAT_FONT.render("Gen: "+str(GEN), 1, BLACK)
    world.win.blit(text, (world.win_width-text.get_width() - 10, 50))

    py.display.update()
    world.win.blit(bg, (0,0))       #blit dello sfondo subito dopo l'update così se ho delle draw prima della draw_win non vengono coperte dallo sfondo
'''
def car_sim(ga_mlp):
    global world, road,GEN
    t = 0
    #FPS=10

    clock = py.time.Clock()

    run = True
    fitness = 0

    car = Car(0, 0, 0)
    clock.tick(1)

    (xb, yb) = (0, 0)
    i = 0

    while run:

        t += 1

        world.updateScore(0)

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False
                py.quit()
                quit()

        input = car.getInputs(world, road)
        input.append(car.vel/MAX_VEL)
        nn_in=np.array(input).reshape(1, -1)
        car.commands = ga_mlp.predict(nn_in).tolist()[0]

        y_old = car.y
        (x, y) = car.move(road,t)
        if t>10 and (car.detectCollision(road) or y > y>y_old or car.vel < 0.1): #il t serve a evitare di eliminare macchine nei primi tot frame (nei primi frame getCollision() restituisce sempre true)
            return fitness
        else:
            fitness += -(y - y_old)/100 + car.vel*SCORE_VEL_MULTIPLIER
            i += 1

        if y < yb:
            (xb, yb) = (x, y)

        road.update(world)
        draw_win([car], road, world)
'''

def car_sim(population):
    global GEN

    t = 0

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    world.win.blit(bg, (0,0))

    road = Road(world)
    clock = py.time.Clock()

    cars=[(Car(0, 0, 0)) for _ in range(len(population))]
    pop_fitness = [0 for i in range(len(population))]

    run = True
    while run:
        t += 1
        clock.tick(FPS)
        world.updateScore(0)

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False
                py.quit()
                quit()

        (xb, yb) = (0,0)
        i = 0
        while(i < len(cars)):
            car = cars[i]

            input = car.getInputs(world, road)
            input.append(car.vel/MAX_VEL)
            nn_in = np.array(input).reshape(1, -1)
            ga_mlp = generate_network(population[i])
            car.commands = ga_mlp.predict(nn_in).tolist()[0]

            y_old = car.y
            (x, y) = car.move(road,t)

            if t>10 and (car.detectCollision(road) or y > world.getBestCarPos()[1] + BAD_GENOME_TRESHOLD or y>y_old or car.vel < 0.1): #il t serve a evitare di eliminare macchine nei primi tot frame (nei primi frame getCollision() restituisce sempre true)
                pop_fitness[i] -= 1
                cars.pop(i)
            else:
                pop_fitness[i] += -(y - y_old)/100 + car.vel*SCORE_VEL_MULTIPLIER
                if(pop_fitness[i] > world.getScore()):
                    world.updateScore(pop_fitness[i])
                    world.bestInputs = input
                    world.bestCommands = car.commands
                i += 1

            if y < yb:
                (xb, yb) = (x, y)


        if len(cars) == 0:
            run = False
            break

        world.updateBestCarPos((xb, yb))
        road.update(world)
        draw_win(cars, road, world)

    return pop_fitness

def generate_ga(ga_config):
    global max_neurons, max_weight_value, min_weight_value ,genetic_model, input_size, output_size, genetic_model_precision
    global last_fitness, nn_n_gen, w_n_gen, num_genes, gene_space, num_generations
    global sol_per_pop , num_parents_mating , parent_selection_type, keep_parents , k_tournament , keep_elitism
    global crossover_type , crossover_probability , mutation_type , mutation_probability, mutation_by_replacement
    global allow_duplicate_genes , verbose, time_delay, save_solutions, save_best_solutions
    global max_neurons, max_weight_value ,min_weight_value, genetic_model,input_size,output_size,genetic_model_precision

    max_neurons = ga_config['max_neurons']
    genetic_model = ga_config['genetic_model']
    input_size = ga_config['input_size']
    output_size = ga_config['output_size']
    max_weight_value = ga_config['max_weight_value']
    min_weight_value = ga_config['min_weight_value']
    num_generations = ga_config["num_generations"]
    sol_per_pop = ga_config["sol_per_pop"]
    num_parents_mating = ga_config["num_parents_mating"]
    parent_selection_type = ga_config["parent_selection_type"]
    keep_parents = ga_config["keep_parents"]
    k_tournament = ga_config["K_tournament"]
    keep_elitism = ga_config["keep_elitism"]
    crossover_type = ga_config["crossover_type"]
    crossover_probability = ga_config["crossover_probability"]
    mutation_type = ga_config["mutation_type"]
    mutation_probability = ga_config["mutation_probability"]
    mutation_by_replacement = ga_config["mutation_by_replacement"]
    allow_duplicate_genes = ga_config["allow_duplicate_genes"]
    verbose = ga_config["verbose"]
    time_delay = ga_config["time_delay"]
    save_solutions = ga_config["save_solutions"]
    save_best_solutions = ga_config["save_best_solutions"]

    last_fitness = 0

    # num of genes used to represent the number of neurons in binary
    nn_n_gen = int(math.log(max_neurons, 2))

    if genetic_model:
        w_n_gen = math.ceil(math.log(max_weight_value / genetic_model_precision, 2)) + 1
        # num_genes = neuron number + neurons weights and bias + output bias
        num_genes = nn_n_gen + (
                    input_size + 1 + output_size) * max_neurons * w_n_gen + w_n_gen
        gene_space = [[0, 1] for _ in range(num_genes)]

    else:
        w_n_gen = 1
        # num_genes = neuron number + neurons weights and bias + output bias
        num_genes = nn_n_gen + (input_size + 1 + output_size) * max_neurons + 1
        weights_space = np.linspace(0, max_weight_value + abs(min_weight_value), 1000).astype(
            'float64') + min_weight_value
        gene_space = [[0, 1] for _ in range(nn_n_gen)] + [weights_space.tolist() for _ in
                                                                    range(num_genes - nn_n_gen)]


    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           K_tournament=k_tournament,
                           parent_selection_type=parent_selection_type,
                           crossover_probability=crossover_probability,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           mutation_by_replacement=mutation_by_replacement,
                           # mutation_num_genes=mutation_num_genes,
                           # mutation_percent_genes=mutation_percent_genes,
                           keep_parents=keep_parents,
                           keep_elitism=keep_elitism,
                           num_genes=num_genes,
                           # gene_type=gene_type,
                           gene_space=gene_space,
                           init_range_low=min_weight_value,
                           init_range_high=max_weight_value,
                           fitness_func=fitness_func,
                           on_start=callback_start,
                           on_cal_pop_fitness=on_cal_pop_fitness,
                           # on_fitness=callback_fitness,
                           # on_parents=callback_parents,
                           # on_crossover=callback_crossover,
                           # on_mutation=callback_mutation,
                           on_generation=callback_generation,
                           # on_stop=callback_stop,
                           allow_duplicate_genes=allow_duplicate_genes,
                           save_solutions=save_solutions,
                           save_best_solutions=save_best_solutions,
                           suppress_warnings=True,
                           parallel_processing=None)
    return ga_instance