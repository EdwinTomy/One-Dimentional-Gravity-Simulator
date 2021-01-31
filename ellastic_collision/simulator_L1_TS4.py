#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Tue Oct 20 14:47:27 2020@author: edwintomy"""from scipy.constants import G as Gimport numpy as npimport matplotlib.pyplot as pltfrom math import expimport pandas as pdfrom sklearn.linear_model import LinearRegressionfrom sklearn import metrics#%%upper_border = 5time_step = 0.0001soften = 10 #exponente^(-r/lamba)lam = 0.1total_test = 1000mass_base = 10000000000border = 5class Particle():    def __init__(self, mass, vel, x_pos):        self.mass = mass        self.vel = vel        self.x_pos = x_pos            def potential_U(self, other):        return -abs(G * self.mass * other.mass / (other.x_pos - self.x_pos))        def kinetic_E(self):        return self.mass * self.vel * self.vel / 2        def lambda_F(r):    #print(1 - exp(-abs(r)/lam))    return exp(-abs(r)/lam)        # Acceleration from gravitational force       def acc(particle, other, soften = soften):        #force = G * particle.mass * other_particle.mass/((other.x_pos - particle.x_pos)**2 + soften)    r = other.x_pos - particle.x_pos    force = G * particle.mass * other.mass/((r**2) + lambda_F(r))    force *= np.sign(other.x_pos - particle.x_pos)    return force/particle.massdef ellastic_collision(a, b):    vel_a = (((a.mass - b.mass)/(a.mass + b.mass))*a.vel)+(((2*b.mass)/(a.mass + b.mass))*b.vel)    vel_b = (((2*a.mass)/(a.mass + b.mass))*a.vel)-(((a.mass - b.mass)/(a.mass + b.mass))*b.vel)    a.vel = vel_a    b.vel = vel_b    def inellastic_collision(a, b):    vel = (a.mass*a.vel + b.mass*b.vel)/(a.mass + b.mass)    a.vel = vel    b.vel = veldef iteration(a, b, phantom = True, ellastic = True, step = time_step):        # Kinematic equations    vel_a = a.vel + acc(a, b) * step    vel_b = b.vel + acc(b, a) * step    pos_a = a.x_pos + step * (vel_a + a.vel)/2    pos_b = b.x_pos + step * (vel_b + b.vel)/2            # Collision    if (a.x_pos > b.x_pos) != (pos_a > pos_b):        if not phantom:            if ellastic:                ellastic_collision(a, b)                return #collision             else:                inellastic_collision(a, b)                a.x_pos = pos_a                b.x_pos = pos_b                return #collision                              a.vel = vel_a    b.vel = vel_b    a.x_pos = pos_a    b.x_pos = pos_b       #%% Perfectly Ellastica = Particle(mass_base + mass_base * 4 * np.random.rand(),                  2 * np.random.rand() - 1,                 np.random.rand() * 5)b = Particle(mass_base + mass_base * 4 * np.random.rand(),                     2 * np.random.rand() - 1,                    np.random.rand() * 5)time = 100000arr = np.zeros((time, 2, 4))# Index 0: Which time mark# Index 1: Which particle # Index 2: Which particle characteristic (x_pos, vel, potential_U, kinetic_E)for i in range(time):    #print('---------',i)    arr[i][0][0] = a.x_pos    arr[i][0][1] = a.vel    arr[i][0][2] = a.potential_U(b)    arr[i][0][3] = a.kinetic_E()        arr[i][1][0] = b.x_pos    arr[i][1][1] = b.vel    arr[i][1][2] = b.potential_U(a)    arr[i][1][3] = b.kinetic_E()        iteration(a, b, False)        plt.plot(arr[:, 0, 0])plt.plot(arr[:, 1, 0])plt.title('Particles Path: Ellastic Collision')plt.xlabel('time (100 microseconds)')plt.ylabel('distance (meters)')plt.legend(['a', 'b'])plt.show()#%% Ellastic energyplt.plot(arr[:, 0, 2] + arr[:, 0, 3] + arr[:, 1, 2] + arr[:, 1, 3])plt.title('System energy: Ellastic Collision')plt.ylim(-10000000000, 10000000000)plt.xlabel('time (100 microseconds)')plt.ylabel('energy (joules)')plt.legend(['a', 'b'])plt.show()plt.plot(arr[:, 0, 2])plt.plot(arr[:, 0, 3])plt.plot(arr[:, 1, 2])plt.plot(arr[:, 1, 3])plt.title('Particle energy: Ellastic')plt.xlabel('time (100 microseconds)')plt.ylabel('energy (joules))')plt.legend(['a potential', 'a kinetic', 'b potential', 'b kinetic'])plt.show()#%% Lambda nums = []for i in range(1, 10000000):    nums.append(i/1000000)  lambdas = []for num in nums:    lambdas.append(lambda_F(num))    plt.plot(nums, lambdas)plt.title('Lambda')plt.xlabel('distance')plt.ylabel('lambda')plt.legend([lam])plt.show()#%%def fake_F(r):    return 1/rdef lambda_Force(r, fake_lambda):    return 1/(r  + fake_lambda)fake_forces = []lambda_forces = []for i in range(len(nums)):    fake_forces.append(fake_F(nums[i]))    lambda_forces.append(lambda_Force(nums[i], lambdas[i]))plt.plot(nums, fake_forces)plt.plot(nums, lambda_forces)plt.title('Force with Pseudopotential')plt.xlabel('distance')plt.ylabel('force')plt.legend(['Normal force', 'With psedudo'])plt.ylim(0, 20)plt.show()#%%## Data Capturetotal_train = 9000total_test = 1000mass_base = 10000000000# ellastic# Index 0 is simulation number# Index 1 is particle # Index 3 is particle's properties # 0 = initial position# 1 = initial velocity # 2 = mass# 3 = final position# 4 = final velocity train = np.zeros((total_train, 2, 5))for i in range(total_train):        a = Particle(mass_base + mass_base * 4 * np.random.rand(),                  -1 + 2 * np.random.rand(),                 np.random.rand() * 5)        b = Particle(mass_base + mass_base * 4 * np.random.rand(),                  -1 + 2 * np.random.rand(),                 np.random.rand() * 5)        train[i][0][0] = a.x_pos    train[i][0][1] = a.vel    train[i][0][2] = a.mass    train[i][1][0] = b.x_pos    train[i][1][1] = b.vel    train[i][1][2] = b.mass         for j in range(100000):        iteration(a, b, False)            train[i][0][3] = a.x_pos    train[i][0][4] = a.vel    train[i][1][3] = b.x_pos    train[i][1][4] = b.veldata = {'initial position of a': train[:,0,0],        'initial position of b': train[:,1,0],        'initial velocity of a': train[:,0,1],        'initial velocity of b': train[:,1,1],        'mass of a': train[:,0,2],        'mass of b': train[:,1,2],        'final position of a': train[:,0,3],        'final position of b': train[:,1,3],        'final velocity of a': train[:,0,4],        'final velocity of b': train[:,1,4]}df = pd.DataFrame(data, columns = ['initial position of a',        'initial position of b',        'initial velocity of a',        'initial velocity of b',        'mass of a',        'mass of b',        'final position of a',        'final position of b',        'final velocity of a',        'final velocity of b'])df.to_csv('/Users/edwintomy/One Dimensional Gravity Simulator/ellastic_collision/L1_TS4_train.csv')test = np.zeros((total_test, 2, 5))for i in range(total_test):        a = Particle(mass_base + mass_base * 4 * np.random.rand(),                  -1 + 2 * np.random.rand(),                 np.random.rand() * 5)    b = Particle(mass_base + mass_base * 4 * np.random.rand(),                  -1 + 2 * np.random.rand(),                 np.random.rand() * 5)        test[i][0][0] = a.x_pos    test[i][0][1] = a.vel    test[i][0][2] = a.mass    test[i][1][0] = b.x_pos    test[i][1][1] = b.vel    test[i][1][2] = b.mass         for j in range(100000):        iteration(a, b, False)            test[i][0][3] = a.x_pos    test[i][0][4] = a.vel    test[i][1][3] = b.x_pos    test[i][1][4] = b.veldata_test = {'initial position of a': test[:,0,0],        'initial position of b': test[:,1,0],        'initial velocity of a': test[:,0,1],        'initial velocity of b': test[:,1,1],        'mass of a': test[:,0,2],        'mass of b': test[:,1,2],        'final position of a': test[:,0,3],        'final position of b': test[:,1,3],        'final velocity of a': test[:,0,4],        'final velocity of b': test[:,1,4]}df_test = pd.DataFrame(data_test, columns = ['initial position of a',        'initial position of b',        'initial velocity of a',        'initial velocity of b',        'mass of a',        'mass of b',        'final position of a',        'final position of b',        'final velocity of a',        'final velocity of b'])df_test.to_csv('/Users/edwintomy/One Dimensional Gravity Simulator/ellastic_collision/L1_TS4_test.csv')#%%df_train = pd.read_csv('ellastic_collision/L1_TS4_train.csv')df_test = pd.read_csv('ellastic_collision/L1_TS4_test.csv')x = df_train[['initial position of a',        'initial position of b',        'initial velocity of a',        'initial velocity of b',        'mass of a',        'mass of b']]y = df_train['final position of a']x_test = df_test[['initial position of a',        'initial position of b',        'initial velocity of a',        'initial velocity of b',        'mass of a',        'mass of b']]y_test = df_test['final position of a']linear_regression = LinearRegression()linear_regression.fit(x, y)y_pred = linear_regression.predict(x_test)df_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))df_comp.sort_values(by = ['Actual'], inplace = True)df_comp = df_comp.reset_index(drop=True)plt.scatter(x = df_comp.index, y = df_comp.Predicted, color='blue')         plt.scatter(x = df_comp.index, y = df_comp.Actual, color='black')plt.show()#%%    