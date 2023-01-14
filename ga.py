import numpy as np
from pygad import pygad
import math
import time
from scipy.special import expit
from car_sim.car import Car
from car_sim.road import Road
from car_sim.world import World
from car_sim.config_variables import *

py.font.init()
bg = py.Surface((WIN_WIDTH, WIN_HEIGHT))
bg.fill(GRAY)

'''
----------------------------
 NN_nk | ... | NN_n0 | w00 | w01 | ...
----------------------------
'''

class MLP:
    def __init__(self, coefs_h, intercepts_h, coefs_o, intercepts_o):
        self.coefs_h = coefs_h
        self.intercepts_h = intercepts_h
        self.coefs_o = coefs_o
        self.intercepts_o = intercepts_o

        ''' MLP weights/bias distribution 
        (input, nn_n) -> in-hl w
        (nn_n,)       -> hl b
        (nn_n, out)   -> hl-out w
        (out,)        -> out b
        '''

    def predict(self, nn_input):
        global ga_config
        h1 = np.dot(self.coefs_h.T, nn_input.T)+self.intercepts_h.reshape(self.intercepts_h.shape[0], -1)
        if ga_config['hidden_layer_activation']=='identity':
            a1 = h1
        elif ga_config['hidden_layer_activation'] == 'logistic':
            a1 = expit(h1)
        elif ga_config['hidden_layer_activation'] == 'tanh':
            a1 = np.tanh(h1)
        ho = np.dot(a1.T, self.coefs_o)+self.intercepts_o.reshape(-1, self.intercepts_o.shape[0])

        if ga_config['output_layer_activation'] == 'identity':
            ao = ho
        elif ga_config['output_layer_activation'] == 'logistic':
            ao = expit(ho)
        elif ga_config['output_layer_activation'] == 'tanh':
            ao = np.tanh(ho)

        return ao

def bit_decoding(n, signed):
    global ga_config

    n = n.astype('int8')
    d = 0
    if signed:
        sign = np.sign(0.5 - n[0])
        n = n[1:]
    else:
        sign = 1
    if ga_config['bit_encoding_type'] == 'binary':
        d = n.dot(2 ** np.arange(n.size)[::-1])
    elif ga_config['bit_encoding_type'] == 'gray':
        d = int(''.join(map(str,n.tolist())), 2)
        m = d >> 1
        while m:
            d ^= m
            m >>= 1
    return sign*d


def neurons_number_calc(bits):
    return bit_decoding(bits, False) + 1


def weights_number_calc(genes):  # genetic mode (binary -> decimal)
    global ga_config, w_n_gen

    weights_temp = []
    for i in range(0, len(genes), w_n_gen):
        w = bit_decoding(genes[i + 1:i + w_n_gen], True) * ga_config['genetic_model_precision']
        weights_temp.append(w)
    return weights_temp


def generate_network(genes):
    global ga_config, nn_n_gen, w_n_gen

    max_neurons = ga_config['max_neurons']
    input_size = ga_config['input_size']
    output_size = ga_config['output_size']

    neurons = neurons_number_calc(genes[:nn_n_gen])

    ''' MLP weights/bias distribution 
    (input, nn_n) -> in-hl w
    (nn_n,)       -> hl b
    (nn_n, out)   -> hl-out w
    (out,)        -> out b
    '''

    # Parse weights
    if ga_config['genetic_model']:
        g = weights_number_calc(genes[nn_n_gen:nn_n_gen + input_size * neurons * w_n_gen])
        coefs_0 = np.array(g).reshape(-1, neurons)
        g = weights_number_calc(
            genes[nn_n_gen + input_size * max_neurons * w_n_gen:
                  nn_n_gen + (input_size * max_neurons + neurons) * w_n_gen])
        intercepts_0 = np.array(g)
        g = weights_number_calc(
            genes[nn_n_gen + (input_size + 1) * max_neurons * w_n_gen:
                  nn_n_gen + ((input_size + 1) * max_neurons + output_size*neurons) * w_n_gen])
        coefs_1 = np.array(g).reshape(neurons, -1)
        g = weights_number_calc(genes[-w_n_gen:])
        intercepts_1 = np.array(g)

    else:
        coefs_0 = np.array(genes[nn_n_gen:nn_n_gen + input_size * neurons]).reshape(-1, neurons)
        intercepts_0 = np.array(genes[nn_n_gen + input_size * max_neurons:nn_n_gen +
                                               input_size * max_neurons + neurons])
        coefs_1 = np.array(genes[nn_n_gen + (input_size+1) * max_neurons:nn_n_gen +
                            (input_size+1) * max_neurons + output_size*neurons]).reshape(neurons, -1)
        intercepts_1 = np.array(genes[-output_size:])

    ga_mlp = MLP(coefs_0, intercepts_0, coefs_1, intercepts_1)

    return ga_mlp

def on_cal_pop_fitness(population):
    return car_sim(population)

# To avoid pygad errors, bypassed with on_cal_pop_fitness
def fitness_func(solution, solution_idx):
    return 0

def ga_run_summary(ga_instance):
    global ga_config

    print('Genetic algorithm MLP training')
    print('Number of generations:   {num_generations}'.format(num_generations=ga_config['num_generations']))
    print('Population size:         {pop_size}'.format(pop_size=ga_instance.pop_size[0]))
    print('Number of genes:         {genes}'.format(genes=ga_instance.pop_size[1]))
    print('Mutation type:           {mutation_type}'.format(mutation_type=ga_config['mutation_type']))
    print('Mutation probability:    {mutation_probability}'.format(mutation_probability=ga_config['mutation_probability']))
    print('Crossover type:          {crossover_type}'.format(crossover_type=ga_config['crossover_type']))
    print('Crossover probability:   {crossover_probability}'.format(crossover_probability=ga_config['crossover_probability']))
    print('\n')

def callback_start(ga_instance):
    global last_fitness, world, road, GEN

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    world.win.blit(bg, (0, 0))
    road = Road(world)
    GEN = 1
    last_fitness = 0
    ga_run_summary(ga_instance)


def callback_generation(ga_instance):
    global ga_config, nn_n_gen, last_fitness, road, GEN

    if ga_instance.generations_completed % ga_config['verbose'] == 0:
        neurons = neurons_number_calc(ga_instance.best_solutions[-1][:nn_n_gen])
        print('Generation {generation}'.format(generation=ga_instance.generations_completed))
        print('Last gen parents:        {par_id}'.format(par_id=ga_instance.last_generation_parents_indices))
        print('Best fitness:            {fitness}'.format(fitness=round(ga_instance.best_solutions_fitness[-1],2)))
        print('Best number of neurons:  {neurons}'.format(neurons=neurons))
        print('Best genes:              {genes}'.format(genes=np.around(ga_instance.best_solutions[-1], decimals=2)))
        if ga_config['verbose'] == 1:
            print('Best genes change:       {change}'.format(change=np.around(ga_instance.best_solutions[-1] - last_fitness, decimals=2)))
            last_fitness = ga_instance.best_solutions[-1]
        print('\n')
    time.sleep(ga_config['time_delay'])
    GEN = ga_instance.generations_completed + 1


def draw_win(cars, road, world):     #x e y sono le coordinate della macchina migliore
    global GEN
    road.draw(world)
    for car in cars:
        car.draw(world)

    text = STAT_FONT.render("Best Car Fitness: "+str(int(world.getScore())), 1, BLACK)
    world.win.blit(text, (world.win_width-text.get_width() - 10, 10))
    text = STAT_FONT.render("Gen: "+str(GEN), 1, BLACK)
    world.win.blit(text, (world.win_width-text.get_width() - 10, 50))

    py.display.update()
    world.win.blit(bg, (0,0))       #blit dello sfondo subito dopo l'update così se ho delle draw prima della draw_win non vengono coperte dallo sfondo

def car_sim(population):
    global GEN, world, road
    t = 0

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    world.win.blit(bg, (0,0))

    road = Road(world)
    clock = py.time.Clock()

    cars = [(Car(0, 0, 0)) for _ in range(len(population))]
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
        while i < len(cars):
            car = cars[i]

            car_input = car.getInputs(world, road)
            car_input.append(car.vel/MAX_VEL)
            nn_input = np.array(car_input).reshape(1, -1)
            ga_mlp = generate_network(population[i])
            car.commands = ga_mlp.predict(nn_input).tolist()[0]
            #print(car.commands)
            y_old = car.y
            (x, y) = car.move(road, t)

            if t > 10 and (car.detectCollision(road)  or y > world.getBestCarPos()[1] + BAD_GENOME_TRESHOLD or y>y_old or car.vel < 0.1 or pop_fitness[i] >= 1+ga_config['max_fitness_stop_criteria']): #il t serve a evitare di eliminare macchine nei primi tot frame (nei primi frame getCollision() restituisce sempre true)
                pop_fitness[i] -= 1
                cars.pop(i)
            else:
                pop_fitness[i] += -(y - y_old)/100 + car.vel*SCORE_VEL_MULTIPLIER
                if pop_fitness[i] > world.getScore():
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


def generate_ga(config):
    global last_fitness, ga_config, nn_n_gen, w_n_gen, num_genes

    ga_config = config

    last_fitness = 0

    # num of genes used to represent the number of neurons in binary
    nn_n_gen = int(math.log(ga_config['max_neurons'], 2))

    if ga_config['genetic_model']:
        w_n_gen = math.ceil(math.log(ga_config['max_weight_value'] / ga_config['genetic_model_precision'], 2)) + 1
        # num_genes = neuron number + neurons weights and bias + output bias
        num_genes = nn_n_gen + (ga_config['input_size'] + 1 + ga_config['output_size']) \
                    * ga_config['max_neurons'] * w_n_gen + w_n_gen
        gene_space = [[0, 1] for _ in range(num_genes)]

    else:
        # w_n_gen = 1
        # num_genes = neuron number + neurons weights and bias + output bias
        num_genes = nn_n_gen + (ga_config['input_size'] + 1 + ga_config['output_size']) * ga_config['max_neurons'] + 1
        weights_space = np.linspace(0, ga_config['max_weight_value'] +
                                    abs(ga_config['min_weight_value']), 1000).astype('float64') + ga_config['min_weight_value']

        gene_space = [[0, 1] for _ in range(nn_n_gen)] + [weights_space.tolist() for _ in
                                                                    range(num_genes - nn_n_gen)]

    stop_criteria = ['reach_'+str(ga_config['max_fitness_stop_criteria']),
                     'saturate_'+str(ga_config['fitness_saturate_stop_criteria'])]

    ga_instance = pygad.GA(num_generations=ga_config['num_generations'],
                           num_parents_mating=ga_config['num_parents_mating'],
                           sol_per_pop=ga_config['sol_per_pop'],
                           K_tournament=ga_config['k_tournament'],
                           parent_selection_type=ga_config['parent_selection_type'],
                           crossover_probability=ga_config['crossover_probability'],
                           crossover_type=ga_config['crossover_type'],
                           mutation_type=ga_config['mutation_type'],
                           mutation_probability=ga_config['mutation_probability'],
                           mutation_by_replacement=ga_config['mutation_by_replacement'],
                           # mutation_num_genes=ga_config['mutation_num_genes'],
                           # mutation_percent_genes=ga_config['mutation_percent_genes'],
                           keep_parents=ga_config['keep_parents'],
                           keep_elitism=ga_config['keep_elitism'],
                           num_genes=num_genes,
                           # gene_type=gene_type,
                           gene_space=gene_space,
                           init_range_low=ga_config['min_weight_value'],
                           init_range_high=ga_config['max_weight_value'],
                           fitness_func=fitness_func,
                           on_start=callback_start,
                           on_cal_pop_fitness=on_cal_pop_fitness,
                           # on_fitness=callback_fitness,
                           # on_parents=callback_parents,
                           # on_crossover=callback_crossover,
                           # on_mutation=callback_mutation,
                           on_generation=callback_generation,
                           # on_stop=callback_stop,
                           # callback_generation = stop_exectuion, # return "stop"
                           allow_duplicate_genes=ga_config['allow_duplicate_genes'],
                           save_solutions=ga_config['save_solutions'],
                           save_best_solutions=ga_config['save_best_solutions'],
                           stop_criteria=stop_criteria,
                           suppress_warnings=True,
                           parallel_processing=None)
    return ga_instance
