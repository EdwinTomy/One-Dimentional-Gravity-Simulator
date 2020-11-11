#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:47:27 2020

@author: edwintomy
"""

from scipy.constants import G as G
import numpy as np
import matplotlib.pyplot as plt
from math import exp


#%%

upper_border = 5
time_step = 0.0001
soften = 1 #exponente^(-r/lamba)
lam = 0.1

class Particle():
    def __init__(self, mass, vel, x_pos, rad = 0.1):
        self.mass = mass
        self.vel = vel
        self.x_pos = x_pos
        self.rad = rad
        
    def potential_U(self, other):
        return - abs(G * self.mass * other.mass / (other.x_pos - self.x_pos))
    
    def kinetic_E(self):
        return self.mass * self.vel * self.vel / 2

# Bounds the particles inside the border
def bound(particle, upper_border = upper_border, lower_border = 0):
    
    if particle.x_pos + particle.rad > upper_border:
        particle.vel *= -1
        particle.x_pos = 2*upper_border - particle.x_pos
        
    if particle.x_pos - particle.rad < lower_border:
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
    if not phantom:
        if (((a.x_pos + a.rad) > (b.x_pos - b.rad) != (pos_a + a.rad) > (pos_b - b.rad)) or
            ((a.x_pos - a.rad) > (b.x_pos + b.rad) != (pos_a - a.rad) > (pos_b + b.rad))):
            if ellastic:
                ellastic_collision(a, b)
                bound(a)
                bound(b)
                return
            else:
                inellastic_collision(a, b)
                a.x_pos = pos_a
                b.x_pos = pos_b
                bound(a)
                bound(b)
                return
                 
    a.vel = vel_a
    b.vel = vel_b
    a.x_pos = pos_a
    b.x_pos = pos_b
    
    # Bounds the particles inside the border
    bound(a)
    bound(b)
    
#%% Phantom

# masses of 10 billion kg
a = Particle(10000000000, 0, 1.5)
b = Particle(10000000000, 0, 3.5)

time_phantom = 10000
arr_phantom = np.zeros((time_phantom, 2, 4))

# Index 0: Which time mark
# Index 1: Which particle 
# Index 2: Which particle characteristic (x_pos, vel, potential_U, kinetic_E)

for i in range(time_phantom):
    arr_phantom[i][0][0] = a.x_pos
    arr_phantom[i][0][1] = a.vel
    arr_phantom[i][0][2] = a.potential_U(b)
    arr_phantom[i][0][3] = a.kinetic_E()
    
    arr_phantom[i][1][0] = b.x_pos
    arr_phantom[i][1][1] = b.vel
    arr_phantom[i][1][2] = b.potential_U(a)
    arr_phantom[i][1][3] = b.kinetic_E()
    
    iteration(a, b)
    

plt.plot(arr_phantom[:, 0, 0])
plt.plot(arr_phantom[:, 1, 0])
plt.title('Particles Path: Phantom')
plt.xlabel('time (miliseconds)')
plt.ylabel('distance (meters)')
plt.legend(['a', 'b'])
plt.show()

#%% Phantom energy

plt.plot(arr_phantom[:, 0, 2] + arr_phantom[:, 0, 3] + arr_phantom[:, 1, 2] + arr_phantom[:, 1, 3])
plt.title('System energy: Phantom')
plt.xlabel('time (miliseconds)')
plt.ylabel('energy (joules)')
plt.legend(['a', 'b'])
plt.show()

plt.plot(arr_phantom[:, 0, 2])
plt.plot(arr_phantom[:, 0, 3])
plt.plot(arr_phantom[:, 1, 2])
plt.plot(arr_phantom[:, 1, 3])
plt.title('Particle energy: Phantom')
plt.xlabel('time (miliseconds)')
plt.ylabel('energy (joules))')
plt.legend(['a potential', 'a kinetic', 'b potential', 'b kinetic'])
plt.show()


#%% Perfectly Ellastic

a = Particle(10000000000, 0, 1.5)
b = Particle(10000000000, 0, 3.5)

time_ellastic = 1000000
arr_ellastic = np.zeros((time_ellastic, 2, 4))

# Index 0: Which time mark
# Index 1: Which particle 
# Index 2: Which particle characteristic (x_pos, vel, potential_U, kinetic_E)

for i in range(time_ellastic):
    #print('---------',i)
    arr_ellastic[i][0][0] = a.x_pos
    arr_ellastic[i][0][1] = a.vel
    arr_ellastic[i][0][2] = a.potential_U(b)
    arr_ellastic[i][0][3] = a.kinetic_E()
    
    arr_ellastic[i][1][0] = b.x_pos
    arr_ellastic[i][1][1] = b.vel
    arr_ellastic[i][1][2] = b.potential_U(a)
    arr_ellastic[i][1][3] = b.kinetic_E()
    
    iteration(a, b, False)
    
    
plt.plot(arr_ellastic[:, 0, 0])
plt.plot(arr_ellastic[:, 1, 0])
plt.title('Particles Path: Ellastic Collision')
plt.xlabel('time (miliseconds)')
plt.ylabel('distance (meters)')
plt.legend(['a', 'b'])
plt.show()


#%% Ellastic energy

plt.plot(arr_ellastic[:, 0, 2] + arr_ellastic[:, 0, 3] + arr_ellastic[:, 1, 2] + arr_ellastic[:, 1, 3])
plt.title('System energy: Ellastic Collision')
plt.xlabel('time (miliseconds)')
plt.ylabel('energy (joules)')
plt.legend(['a', 'b'])
plt.show()

plt.plot(arr_ellastic[:, 0, 2])
plt.plot(arr_ellastic[:, 0, 3])
plt.plot(arr_ellastic[:, 1, 2])
plt.plot(arr_ellastic[:, 1, 3])
plt.title('Particle energy: Ellastic')
plt.xlabel('time (miliseconds)')
plt.ylabel('energy (joules))')
plt.legend(['a potential', 'a kinetic', 'b potential', 'b kinetic'])
plt.show()

#%% Perfectly Inellastic 

a = Particle(10000000000, 0, 1.5)
b = Particle(10000000000, -2, 3.5)

time_inellastic = 3000
arr_inellastic = np.zeros((time_inellastic, 2, 4))

# Index 0: Which time mark
# Index 1: Which particle 
# Index 2: Which particle characteristic (x_pos, vel, potential_U, kinetic_E)

for i in range(time_inellastic):
    arr_inellastic[i][0][0] = a.x_pos
    arr_inellastic[i][0][1] = a.vel
    arr_inellastic[i][0][2] = a.potential_U(b)
    arr_inellastic[i][0][3] = a.kinetic_E()
    
    arr_inellastic[i][1][0] = b.x_pos
    arr_inellastic[i][1][1] = b.vel
    arr_inellastic[i][1][2] = b.potential_U(a)
    arr_inellastic[i][1][3] = b.kinetic_E()
    
    iteration(a, b, False, False)
    
plt.plot(arr_inellastic[:, 0, 0])
plt.plot(arr_inellastic[:, 1, 0])
plt.title('Particles Path: Inellastic Collision')
plt.xlabel('time (miliseconds)')
plt.ylabel('distance (meters)')
plt.legend(['a', 'b'])
plt.show()

#%% Inellastic energy

plt.plot(arr_inellastic[:, 0, 2] + arr_inellastic[:, 0, 3] + arr_inellastic[:, 1, 2] + arr_inellastic[:, 1, 3])
plt.title('System energy: Inellastic Collision')
plt.xlabel('time (miliseconds)')
plt.ylabel('energy (joules)')
plt.legend(['a', 'b'])
plt.show()

plt.plot(arr_inellastic[:, 0, 2])
plt.plot(arr_inellastic[:, 0, 3])
plt.plot(arr_inellastic[:, 1, 2])
plt.plot(arr_inellastic[:, 1, 3])
plt.title('Particle energy: Inellastic')
plt.xlabel('time (miliseconds)')
plt.ylabel('energy (joules))')
plt.legend(['a potential', 'a kinetic', 'b potential', 'b kinetic'])
plt.show()

#%% Lambda 

nums = []
for i in range(1, 10000000):
    nums.append(i/1000000)

  
lambdas = []
for num in nums:
    lambdas.append(lambda_F(num))
    
plt.plot(nums, lambdas)
plt.title('Lambda')
plt.xlabel('distance')
plt.ylabel('lambda')
plt.legend([lam])
plt.show()



#%%

def fake_F(r):
    return 1/r

def lambda_Force(r, fake_lambda):
    return 1/(r  + fake_lambda)

fake_forces = []
lambda_forces = []
for i in range(len(nums)):
    fake_forces.append(fake_F(nums[i]))
    lambda_forces.append(lambda_Force(nums[i], lambdas[i]))

plt.plot(nums, fake_forces)
plt.plot(nums, lambda_forces)

plt.title('Force with Pseudopotential')
plt.xlabel('distance')
plt.ylabel('force')
plt.legend(['Normal force', 'With psedudo'])
plt.ylim(0, 20)
plt.show()

#%%





    