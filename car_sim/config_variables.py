import pygame as py
py.font.init()

def config_load(ga_config):
    global FPS,WIN_WIDTH,WIN_HEIGHT,STARTING_POS,SCORE_VEL_MULTIPLIER,INPUT_NEURONS,OUTPUT_NEURONS
    global CAR_DBG,FRICTION,MAX_VEL,MAX_VEL_REDUCTION,ACC_STRENGHT,TURN_VEL,SENSOR_DISTANCE,SENSORS_INITIAL_ANGLE
    global SENSORS_INITIAL_ANGLE,SENSORS_SEPARATION_ANGLE,SENSORS_NUMBER_BEAMS,BAD_GENOME_TRESHOLD
    global ROAD_DBG,MAX_ANGLE,MAX_DEVIATION,SPACING,NUM_POINTS,SAFE_SPACE,ROAD_WIDTH

    FPS = ga_config['sim_fps']
    WIN_WIDTH = ga_config['sim_win_w']
    WIN_HEIGHT = ga_config['sim_win_h']
    STARTING_POS = (WIN_WIDTH/2, WIN_HEIGHT-ga_config['sim_start_coord'])
    SCORE_VEL_MULTIPLIER = ga_config['speed_bonus']

    INPUT_NEURONS = ga_config['input_size']
    OUTPUT_NEURONS = ga_config['output_size']

    #=================== Car Specs ==================================

    CAR_DBG = ga_config['sensors_visuals']
    FRICTION = ga_config['road_friction']
    MAX_VEL = ga_config['max_vel']
    MAX_VEL_REDUCTION = ga_config['max_vel_reduction']
    ACC_STRENGHT = ga_config['acc_strength']
    TURN_VEL = ga_config['turn_vel']
    SENSOR_DISTANCE = ga_config['sensor_max_range']
    SENSORS_INITIAL_ANGLE = ga_config['sensors_first_beam_angle']
    SENSORS_SEPARATION_ANGLE = ga_config['sensors_beams_separation_angle']
    SENSORS_NUMBER_BEAMS = ga_config['sensors_beams_number_angle']
    BAD_GENOME_TRESHOLD = ga_config['bad_genome_thershold']

    #=================== Road Specs ==================================

    ROAD_DBG = ga_config['road_visuals']
    MAX_ANGLE = ga_config['road_max_angle']
    MAX_DEVIATION = ga_config['road_max_deviation']
    SPACING = ga_config['road_spacing']
    NUM_POINTS = ga_config['road_points_per_segment']
    SAFE_SPACE = SPACING + 50       #buffer space above the screen
    ROAD_WIDTH = ga_config['road_width']

#=================== Display and Colors ==================================

NODE_RADIUS = 20
NODE_SPACING = 5
LAYER_SPACING = 100
CONNECTION_WIDTH = 1

WHITE = (255, 255, 255)
GRAY = (255, 255, 255)
#GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
DARK_RED = (100, 0, 0)
RED_PALE = (250, 200, 200)
DARK_RED_PALE = (150, 100, 100)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 100, 0)
GREEN_PALE = (200, 250, 200)
DARK_GREEN_PALE = (100, 150, 100)
BLUE = (0,0,255)
BLUE_PALE = (200, 200, 255)
DARK_BLUE = (100, 100, 150)

NODE_FONT = py.font.SysFont("arial", 15)
STAT_FONT = py.font.SysFont("arial", 30)


#=================== Constants for internal use ==================================
GEN = 0

#enumerations
ACC_BRAKE = 0
TURN = 1

INPUT = 0
MIDDLE = 1
OUTPUT = 2
