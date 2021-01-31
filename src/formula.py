from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy.constants import G as G
from math import exp


class Particle():
    def __init__(self, mass, vel, x_pos):
        self.mass = mass
        self.vel = vel
        self.x_pos = x_pos
        
    def potential_U(self, other):
        return -abs(G * self.mass * other.mass / (other.x_pos - self.x_pos))
    
    def kinetic_E(self):
        return self.mass * self.vel * self.vel / 2

# Bounds the particles inside the border
def bound(particle, upper_border = upper_border, lower_border = 0):
    
    if particle.x_pos > upper_border:
        particle.vel *= -1
        particle.x_pos = 2*upper_border - particle.x_pos
        
    if particle.x_pos < lower_border:
        particle.vel *= -1
        particle.x_pos = 2*lower_border - particle.x_pos
        
def lambda_F(r):
    #print(1 - exp(-abs(r)/lam))
    return exp(-abs(r)/lam)
        
# Acceleration from gravitational force       
def acc(particle, other, soften = soften):
    
    #force = G * particle.mass * other_particle.mass/((other.x_pos - particle.x_pos)**2 + soften)
    r = other.x_pos - particle.x_pos
    force = G * particle.mass * other.mass/((r**2) + lambda_F(r))

    force *= np.sign(other.x_pos - particle.x_pos)
    return force/particle.mass

def ellastic_collision(a, b):
    vel_a = (((a.mass - b.mass)/(a.mass + b.mass))*a.vel)+(((2*b.mass)/(a.mass + b.mass))*b.vel)
    vel_b = (((2*a.mass)/(a.mass + b.mass))*a.vel)-(((a.mass - b.mass)/(a.mass + b.mass))*b.vel)
    a.vel = vel_a
    b.vel = vel_b
    
def inellastic_collision(a, b):
    vel = (a.mass*a.vel + b.mass*b.vel)/(a.mass + b.mass)
    a.vel = vel
    b.vel = vel

def iteration(a, b, phantom = True, ellastic = True, step = time_step):
    
    # Kinematic equations
    vel_a = a.vel + acc(a, b) * step
    vel_b = b.vel + acc(b, a) * step
    pos_a = a.x_pos + step * (vel_a + a.vel)/2
    pos_b = b.x_pos + step * (vel_b + b.vel)/2
    
    
    # Collision
    if (a.x_pos > b.x_pos) != (pos_a > pos_b):
        if not phantom:
            if ellastic:
                ellastic_collision(a, b)
                bound(a)
                bound(b)
                return #collision 
            else:
                inellastic_collision(a, b)
                a.x_pos = pos_a
                b.x_pos = pos_b
                bound(a)
                bound(b)
                return #collision 
        
                 
    a.vel = vel_a
    b.vel = vel_b
    a.x_pos = pos_a
    b.x_pos = pos_b
    
    # Bounds the particles inside the border
    bound(a)
    bound(b)
    
    return #collision