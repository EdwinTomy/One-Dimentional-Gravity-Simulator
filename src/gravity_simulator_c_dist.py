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
import pandas as pd


#%%


upper_border = 5
time_step = 0.0001
soften = 10 #exponente^(-r/lamba)
lam = 0.1
total_train = 9000
total_test = 1000
mass_base = 10000000000
border = 5

class Particle():
    def __init__(self, mass, vel, x_pos, collided = 100000):
        self.mass = mass
        self.vel = vel
        self.x_pos = x_pos
        self.collided = collided
        
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
            a.collided = 0
            b.collided = 0
            if ellastic:
                ellastic_collision(a, b)
                #bound(a)
                #bound(b)
                return #collision 
            else:
                inellastic_collision(a, b)
                a.x_pos = pos_a
                b.x_pos = pos_b
                #bound(a)
                #bound(b)
                return #collision 
        
                 
    a.vel = vel_a
    b.vel = vel_b
    a.x_pos = pos_a
    b.x_pos = pos_b
    
    # Bounds the particles inside the border
    #bound(a)
    #bound(b)
    
    if abs(a.x_pos - b.x_pos) < a.collided:
        a.collided = abs(pos_a - pos_b)
        b.collided = abs(pos_a - pos_b)
    
    return #collision
    
#%% Phantom

# masses of 10 billion kg
a = Particle(10000000000, 0, 1.5)
b = Particle(10000000000, 0, 1.6)

time_phantom = 100000
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
plt.xlabel('time (100 microseconds)')
plt.ylabel('distance (meters)')
plt.legend(['a', 'b'])
plt.show()

#%% Phantom energy

plt.plot(arr_phantom[:, 0, 2] + arr_phantom[:, 0, 3] + arr_phantom[:, 1, 2] + arr_phantom[:, 1, 3])
plt.ylim(-10000000000, 10000000000)
plt.title('System energy: Phantom')
plt.xlabel('time (100 microseconds)')
plt.ylabel('energy (joules)')
plt.legend(['a', 'b'])
plt.show()

plt.plot(arr_phantom[:, 0, 2])
plt.plot(arr_phantom[:, 0, 3])
plt.plot(arr_phantom[:, 1, 2])
plt.plot(arr_phantom[:, 1, 3])
plt.title('Particle energy: Phantom')
plt.xlabel('time (100 microseconds)')
plt.ylabel('energy (joules))')
plt.legend(['a potential', 'a kinetic', 'b potential', 'b kinetic'])
plt.show()


#%% Perfectly Ellastic

a = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 -1 + 2 * np.random.rand(),
                 2)
b = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 -1 + 2 * np.random.rand(),
                 3)


time_ellastic = 10000
arr_ellastic = np.zeros((time_ellastic, 2, 4))

# Index 0: Which time mark
# Index 1: Which particle 
# Index 2: Which particle characteristic (x_pos, vel, potential_U, kinetic_E)
collision = time_ellastic
for i in range(time_ellastic):
    
    if a.collided==0 and (collision == time_ellastic):
        collision = i
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
    
print(a.collided, collision)
    

    
plt.plot(arr_ellastic[:, 0, 0])
plt.plot(arr_ellastic[:, 1, 0])
plt.title('Particles Path: Ellastic Collision')
plt.xlabel('time (100 microseconds)')
plt.ylabel('distance (meters)')
plt.legend(['a', 'b'])
plt.show()


#%% Ellastic energy

plt.plot(arr_ellastic[:, 0, 2] + arr_ellastic[:, 0, 3] + arr_ellastic[:, 1, 2] + arr_ellastic[:, 1, 3])
plt.title('System energy: Ellastic Collision')
plt.ylim(-10000000000, 10000000000)
plt.xlabel('time (100 microseconds)')
plt.ylabel('energy (joules)')
plt.legend(['a', 'b'])
plt.show()

plt.plot(arr_ellastic[:, 0, 2])
plt.plot(arr_ellastic[:, 0, 3])
plt.plot(arr_ellastic[:, 1, 2])
plt.plot(arr_ellastic[:, 1, 3])
plt.title('Particle energy: Ellastic')
plt.xlabel('time (100 microseconds)')
plt.ylabel('energy (joules))')
plt.legend(['a potential', 'a kinetic', 'b potential', 'b kinetic'])
plt.show()

#%% Perfectly Inellastic 
total_train = 9000
total_test = 1000
mass_base = 10000000000
border = 5

a = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 2 * np.random.rand() - 1,
                 np.random.rand() * 5)

b = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                    2 * np.random.rand() - 1,
                    np.random.rand() * 5)

time_inellastic = 1000
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
    
    if iteration(a, b, False, False):
        print('True')
    
plt.plot(arr_inellastic[:, 0, 0])
plt.plot(arr_inellastic[:, 1, 0])
plt.title('Particles Path: Inellastic Collision')
plt.xlabel('time (100 microseconds)')
plt.ylabel('distance (meters)')
plt.legend(['a', 'b'])
plt.show()

#%% Inellastic energy

plt.plot(arr_inellastic[:, 0, 2] + arr_inellastic[:, 0, 3] + arr_inellastic[:, 1, 2] + arr_inellastic[:, 1, 3])
plt.title('System energy: Inellastic Collision')
plt.xlabel('time (100 mircoseconds)')
plt.ylabel('energy (joules)')
plt.legend(['a', 'b'])
plt.show()

plt.plot(arr_inellastic[:, 0, 2])
plt.plot(arr_inellastic[:, 0, 3])
plt.plot(arr_inellastic[:, 1, 2])
plt.plot(arr_inellastic[:, 1, 3])
plt.title('Particle energy: Inellastic')
plt.xlabel('time (100 microseconds)')
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

## Data Capture

total_train = 9000
total_test = 1000
mass_base = 10000000000
border = 5

# collision_iwhen

# Index 0 is simulation number
# Index 1 is particle 
# Index 3 is particle's properties 
# 0 = initial position
# 1 = initial velocity 
# 2 = mass
# 3 = final position
# 4 = final velocity 

collision_when_train = np.zeros((total_train, 2, 6))
for i in range(total_train):
    
    a = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 -1 + 2 * np.random.rand(),
                 np.random.rand() * 5)
    b = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 -1 + 2 * np.random.rand(),
                 np.random.rand() * 5)
    
    collision_when_train[i][0][0] = a.x_pos
    collision_when_train[i][0][1] = a.vel
    collision_when_train[i][0][2] = a.mass
    collision_when_train[i][1][0] = b.x_pos
    collision_when_train[i][1][1] = b.vel
    collision_when_train[i][1][2] = b.mass
     
    collision = 50000
    for j in range(50000):
        if a.collided and (collision == 50000):
            collision = j
        iteration(a, b, False)
        
    collision_when_train[i][0][3] = a.x_pos
    collision_when_train[i][0][4] = a.vel
    collision_when_train[i][1][3] = b.x_pos
    collision_when_train[i][1][4] = b.vel
    collision_when_train[i][0][5] = collision

data_collision_when = {'initial position of a': collision_when_train[:,0,0],
        'initial position of b': collision_when_train[:,1,0],
        'initial velocity of a': collision_when_train[:,0,1],
        'initial velocity of b': collision_when_train[:,1,1],
        'mass of a': collision_when_train[:,0,2],
        'mass of b': collision_when_train[:,1,2],
        'final position of a': collision_when_train[:,0,3],
        'final position of b': collision_when_train[:,1,3],
        'final velocity of a': collision_when_train[:,0,4],
        'final velocity of b': collision_when_train[:,1,4],
        'collision time' : collision_when_train[:,0,5]}

df_collision_when = pd.DataFrame(data_collision_when, columns = ['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b',
        'final position of a',
        'final position of b',
        'final velocity of a',
        'final velocity of b',
        'collision time'])

df_collision_when.to_csv('/Users/edwintomy/One Dimensional Gravity Simulator/data/collision_when_training.csv')

collision_when_test = np.zeros((total_test, 2, 6))
for i in range(total_test):
    
    a = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 -1 + 2 * np.random.rand(),
                 np.random.rand() * 5)
    b = Particle(mass_base + mass_base * 4 * np.random.rand(), 
                 -1 + 2 * np.random.rand(),
                 np.random.rand() * 5)
    
    collision_when_test[i][0][0] = a.x_pos
    collision_when_test[i][0][1] = a.vel
    collision_when_test[i][0][2] = a.mass
    collision_when_test[i][1][0] = b.x_pos
    collision_when_test[i][1][1] = b.vel
    collision_when_test[i][1][2] = b.mass
     
    collision = 50000
    for j in range(50000):
        if a.collided and (collision == 50000):
            collision = j
        iteration(a, b, False)
        
    collision_when_test[i][0][3] = a.x_pos
    collision_when_test[i][0][4] = a.vel
    collision_when_test[i][1][3] = b.x_pos
    collision_when_test[i][1][4] = b.vel
    collision_when_test[i][0][5] = collision

data_collision_when_test = {'initial position of a': collision_when_test[:,0,0],
        'initial position of b': collision_when_test[:,1,0],
        'initial velocity of a': collision_when_test[:,0,1],
        'initial velocity of b': collision_when_test[:,1,1],
        'mass of a': collision_when_test[:,0,2],
        'mass of b': collision_when_test[:,1,2],
        'final position of a': collision_when_test[:,0,3],
        'final position of b': collision_when_test[:,1,3],
        'final velocity of a': collision_when_test[:,0,4],
        'final velocity of b': collision_when_test[:,1,4],
        'collision time' : collision_when_test[:,0,5]}

df_collision_when_test = pd.DataFrame(data_collision_when_test, columns = ['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b',
        'final position of a',
        'final position of b',
        'final velocity of a',
        'final velocity of b',
        'collision time'])


#%%
df_collision_when_test.to_csv('/Users/edwintomy/One Dimensional Gravity Simulator/data/collision_when_testing.csv')
#%%

# collision_if
# Index 0 is simulation number
# Index 1 is particle 
# Index 3 is particle's properties 
# 0 = initial position
# 1 = initial velocity 
# 2 = mass
# 3 = final position
# 4 = final velocity 

collision_if_train = np.zeros((total_train, 2, 6))
for i in range(total_train):
    
    a = Particle(mass_base, 
                 -1 + 2 * np.random.rand(),
                 3)
    b = Particle(mass_base, 
                 -1 + 2 * np.random.rand(),
                 2)
    
    collision_if_train[i][0][0] = a.x_pos
    collision_if_train[i][0][1] = a.vel
    collision_if_train[i][0][2] = a.mass
    collision_if_train[i][1][0] = b.x_pos
    collision_if_train[i][1][1] = b.vel
    collision_if_train[i][1][2] = b.mass
     
    for j in range(10000):
        iteration(a, b, False)
        
    collision_if_train[i][0][3] = a.x_pos
    collision_if_train[i][0][4] = a.vel
    collision_if_train[i][1][3] = b.x_pos
    collision_if_train[i][1][4] = b.vel
    collision_if_train[i][0][5] = a.collided


data_collision_if = {'initial position of a': collision_if_train[:,0,0],
        'initial position of b': collision_if_train[:,1,0],
        'initial velocity of a': collision_if_train[:,0,1],
        'initial velocity of b': collision_if_train[:,1,1],
        'mass of a': collision_if_train[:,0,2],
        'mass of b': collision_if_train[:,1,2],
        'final position of a': collision_if_train[:,0,3],
        'final position of b': collision_if_train[:,1,3],
        'final velocity of a': collision_if_train[:,0,4],
        'final velocity of b': collision_if_train[:,1,4],
        'if collided':collision_if_train[:,0,5]}

df_collision_if = pd.DataFrame(data_collision_if, columns = ['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b',
        'final position of a',
        'final position of b',
        'final velocity of a',
        'final velocity of b',
        'if collided'])

df_collision_if.to_csv('/Users/edwintomy/One Dimensional Gravity Simulator/data/collision_if_dist_training.csv')

collision_if_test = np.zeros((total_test, 2, 6))
for i in range(total_test):
    
    a = Particle(mass_base, 
                 -1 + 2 * np.random.rand(),
                 3)
    b = Particle(mass_base, 
                 -1 + 2 * np.random.rand(),
                 2)
    
    collision_if_test[i][0][0] = a.x_pos
    collision_if_test[i][0][1] = a.vel
    collision_if_test[i][0][2] = a.mass
    collision_if_test[i][1][0] = b.x_pos
    collision_if_test[i][1][1] = b.vel
    collision_if_test[i][1][2] = b.mass
     
    for j in range(10000):
        iteration(a, b, False)
        
    collision_if_test[i][0][3] = a.x_pos
    collision_if_test[i][0][4] = a.vel
    collision_if_test[i][1][3] = b.x_pos
    collision_if_test[i][1][4] = b.vel
    collision_if_test[i][0][5] = a.collided
    
    #print(a.x_pos, a.vel, b.x_pos, b.vel)

data_collision_if_test = {'initial position of a': collision_if_test[:,0,0],
        'initial position of b': collision_if_test[:,1,0],
        'initial velocity of a': collision_if_test[:,0,1],
        'initial velocity of b': collision_if_test[:,1,1],
        'mass of a': collision_if_test[:,0,2],
        'mass of b': collision_if_test[:,1,2],
        'final position of a': collision_if_test[:,0,3],
        'final position of b': collision_if_test[:,1,3],
        'final velocity of a': collision_if_test[:,0,4],
        'final velocity of b': collision_if_test[:,1,4],
        'if collided': collision_if_test[:,0,5]}

df_collision_if_test = pd.DataFrame(data_collision_if_test, columns = ['initial position of a',
        'initial position of b',
        'initial velocity of a',
        'initial velocity of b',
        'mass of a',
        'mass of b',
        'final position of a',
        'final position of b',
        'final velocity of a',
        'final velocity of b',
        'if collided'])

df_collision_if_test.to_csv('/Users/edwintomy/One Dimensional Gravity Simulator/data/collision_if_dist_testing.csv')

#%%








    