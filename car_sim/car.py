from car_sim.config_variables import *
import pygame as py
import os
from math import *
from random import random
from car_sim.road import *
import numpy as np
from car_sim.vect2d import vect2d


class Car:
    x = 0
    y = 0       #coordinate rispetto al sistema di riferimento globale, la posizione sullo schermo è relativa alla posizione della macchina migliore


    def __init__(self, x, y, turn):
        self.x = x
        self.y = y
        self.rot = turn
        self.rot = 0
        self.vel = MAX_VEL/2
        self.acc = 0
        self.initImgs()
        self.commands = [0,0,0,0]

    def initImgs(self):
        img_names = ["yellow_car.png", "red_car.png", "blu_car.png", "green_car.png"]
        name = img_names[floor(random()*len(img_names))%len(img_names)]                 #prendi a caso una di queste immagini

        self.img = py.transform.rotate(py.transform.scale(py.image.load(os.path.join("car_sim/imgs", name)).convert_alpha(), (120, 69)), -90)
        self.brake_img = py.transform.rotate(py.transform.scale(py.image.load(os.path.join("car_sim/imgs", "brakes.png")).convert_alpha(), (120, 69)), -90)

    def detectCollision(self, road):
        #get mask
        mask = py.mask.from_surface(self.img)
        (width, height) = mask.get_size()
        for v in [road.pointsLeft, road.pointsRight]:
            for p in v:
                x = p.x - self.x + width/2
                y = p.y - self.y + height/2
                try:
                    if mask.get_at((int(x),int(y))):
                        return True
                except IndexError as error:
                    continue
        return False

    def getInputs(self, world, road):         #win serve per disegnare i sensori se DBG = True
        sensors = []
        for k in range(SENSORS_NUMBER_BEAMS):
            sensors.append(SENSOR_DISTANCE)
        sensorsEquations = getSensorEquations(self, world)

        for v in [road.pointsLeft, road.pointsRight]:
            i = road.bottomPointIndex
            while v[i].y > self.y - SENSOR_DISTANCE:
                next_index = getPoint(i+1, NUM_POINTS*road.num_ctrl_points)

                getDistance(world, self, sensors, sensorsEquations, v[i], v[next_index])
                i = next_index

        if CAR_DBG:
            for k,s in enumerate(sensors):
                omega = radians(self.rot + SENSORS_INITIAL_ANGLE+SENSORS_SEPARATION_ANGLE*k)
                dx = s * sin(omega)
                dy = - s * cos(omega)
                #disegna intersezioni dei sensori
                if s < SENSOR_DISTANCE:
                    py.draw.circle(world.win, RED, world.getScreenCoords(self.x+dx, self.y+dy), 6)

        #convert to value between 0 (distance = max) and 1 (distance = 0)
        for s in range(len(sensors)):
            sensors[s] = 1 - sensors[s]/SENSOR_DISTANCE

        return sensors


    def move(self, road, t):
        self.acc = FRICTION

        self.acc = ACC_STRENGHT*self.commands[ACC_BRAKE]
        self.rot += TURN_VEL*self.commands[TURN]

        timeBuffer = 500
        if MAX_VEL_REDUCTION == 1 or t >= timeBuffer:
            max_vel_local = MAX_VEL
        else:
            ratio = MAX_VEL_REDUCTION + (1 - MAX_VEL_REDUCTION)*(t/timeBuffer)
            max_vel_local = MAX_VEL * ratio

        self.vel += self.acc
        if self.vel > max_vel_local:
            self.vel = max_vel_local
        if self.vel < 0:
            self.vel = 0
        self.x = self.x + self.vel * sin(radians(self.rot))
        self.y = self.y - self.vel * cos(radians(self.rot)) #sottraggo perchè l'origine è in alto a sinistra

        #print("coord: ("+str(self.x)+", "+str(self.y)+")   vel: "+str(self.vel)+"   acc: "+str(self.acc) + "    rot: "+str(self.rot))

        return (self.x, self.y)

    def draw(self, world):
        screen_position = world.getScreenCoords(self.x, self.y)
        rotated_img = py.transform.rotate(self.img, -self.rot)
        new_rect = rotated_img.get_rect(center = screen_position)
        world.win.blit(rotated_img, new_rect.topleft)

        if self.commands[ACC_BRAKE] < 0:
            rotated_img = py.transform.rotate(self.brake_img, -self.rot)
            new_rect = rotated_img.get_rect(center = screen_position)
            world.win.blit(rotated_img, new_rect.topleft)

    #======================== LOCAL FUNCTIONS ==========================

def getSensorEquations(self, world):       #restituisce le equazioni delle rette (in variabile y) della macchina in ordine [verticale, diagonale crescente, orizzontale, diagonale decrescente]
    eq = []
    for i in range(SENSORS_NUMBER_BEAMS):
        omega = radians(self.rot + SENSORS_INITIAL_ANGLE+SENSORS_SEPARATION_ANGLE*i)
        dx = SENSOR_DISTANCE * sin(omega)
        dy = - SENSOR_DISTANCE * cos(omega)

        if CAR_DBG:             #disegna linee dei sensori
            py.draw.lines(world.win, GREEN, False, [world.getScreenCoords(self.x+dx, self.y+dy), world.getScreenCoords(self.x, self.y)], 2)

        coef = getSegmentEquation(self, vect2d(x = self.x+dx, y = self.y+dy))
        eq.append(coef)
    return eq

def getSegmentEquation(p, q):          #equazioni in variabile y tra due punti (tenendo conto del sistema di coordinate con y invertito) nella forma generale ax + by + c =  0

    a = p.y - q.y
    b = q.x -p.x
    c = p.x*q.y - q.x*p.y

    return (a,b,c)

def getDistance(world, car, sensors, sensorsEquations, p, q):     #dato il segmento (m,q) calcolo la distanza e la metto nel sensore corrispondente
    (a2,b2,c2) = getSegmentEquation(p, q)
    for i,(a1,b1,c1) in enumerate(sensorsEquations):
        #get intersection between sensor and segment

        if a1!=a2 or b1!=b2:
            d = b1*a2 - a1*b2
            if d == 0:
                continue
            y = (a1*c2 - c1*a2)/d
            x = (c1*b2 - b1*c2)/d
            if (y-p.y)*(y-q.y) > 0 or (x-p.x)*(x-q.x) > 0:        #se l'intersezione non sta tra a e b, vai alla prossima iterazione
                continue
        else:       #rette coincidenti
            (x, y) = (abs(p.x-q.x), abs(p.y-q.y))

        #get distance
        dist = ((car.x - x)**2 + (car.y - y)**2)**0.5

        #inserisci nel sensore nel verso giusto
        omega = car.rot + SENSORS_INITIAL_ANGLE+SENSORS_SEPARATION_ANGLE*i                               #angolo della retta del sensore (e del suo opposto)
        alpha = 90- degrees(atan2(car.y - y, x-car.x))     #angolo rispetto alla verticale (come car.rot)
        if cos(alpha)*cos(omega)*100 + sin(alpha)*sin(omega)*100 > 0:
            if dist < sensors[i]:
                sensors[i] = dist

    #----
