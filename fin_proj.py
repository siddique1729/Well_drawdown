# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:59:33 2022

@author: Ameri Gurley, Jawwad Siddique, Xi Wang

Final Project

City wants to acquire an existing land with 10 wells 
Aquifer has hudraulic conductivity, K = 350 ft/day  (Range 200-700)
Storage coefficient = 0.001 (Range 0.0009-0.0025)
Well life is 30 years

- Drawdown <= 75% well depth by end of life
- Outer edge at a minimum, 4 monitoring wells
- 4 backup wells
- population 30,000-40,000
- mean use per capita 100 gpcd +- 15 gcpd
- meet with 95% efficiency
- 4 production wells with 6 backup wells  
- no more than 6 production wells

Estimate annual cost
"""

import pulp as pp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import math

# set working directory
dir = "C:/Users/User/Downloads"
#dir = "C:/Users/18067/Dropbox (Personal)/CE 5331 - Optimization/MiniProj_03"
os.chdir(dir)

# Upload well data
a = pd.read_csv("Well_Data.csv")
a.columns

# Plot well locations 
Xwell = a.x_Feet
Ywell = a.y_Feet
wellID = a.Well_ID

plt.plot(Xwell, Ywell, "bo")
plt.xlabel("Well Location x (ft)")
plt.ylabel("Well Location y (ft)")
plt.grid()
plt.title("Well Location Plot")
for i, txt in enumerate(wellID):
    plt.annotate(txt, (Xwell[i], Ywell[i]))
plt.show()


# Constants

K = 350 #ft/day  (200-700)
S = 0.001 #(0.0009-0.0025)
Dj = a.Depth_FT*.25 # to limit well depth to 75% just make that the allowable depth?
# Option 2: Dj = 150 which is 75% the shallowest well
rj = a.Radius_in/12 # put radius into feet
r = pd.to_numeric(rj)
# t = time since pumping started... in days? K in ft/day
Q = 3000000/7.48 # gpd converted to ft^3/day
t = 366 #10980 # 30 years of leap year days

# Modified Theis Function

def well_func(r, S, K, t):
   u = ((r*r)*S)/(4*K*r*t)
   eu = 0.57721566
   G = np.exp(-1*eu)
   b = ((2-2*G)/(2*G-G*G))**0.5
   hi = ((1-G)*(G*G-6*G+12))/(3*G*(2-G)*(2-G)*b)
   q = (20/47)*(u**((31/26)**0.5))
   h = (1/(1+(u**(3/2))))+((hi*q)/(1+q))
   
   
   nume = (np.exp(-1*u))*(np.log(1+(G/u)+((1-G)/((h+b*u)*(h+b*u)))))
   deno = G + ((1-G)*(np.exp((-1*u))))
   
   Wu = nume/deno
   
   return Wu

W = well_func(r, S, K, t)

# Define drawdown function
def draw_down(Q, K, Dj, W):
    num = Q*W
    p = np.pi
    den = 4*p*K*Dj
    dd = num/den
    return (dd)

ddown = draw_down(Q, K, Dj, W)

# Calculating distances

dist1 = []
dist2 = []
dist3 = []
dist4 = []

for i in range(0,10,1):
    distt1 = ((Xwell[i]-0)**2 +(Ywell[i]-0)**2)**0.5
    distt2 = ((Xwell[i]-0)**2 +(Ywell[i]-5000)**2)**0.5
    distt3 = ((Xwell[i]-5000)**2 +(Ywell[i]-5000)**2)**0.5
    distt4 = ((Xwell[i]-5000)**2 +(Ywell[i]-0)**2)**0.5
    
    dist1.append(distt1)
    dist2.append(distt2)
    dist3.append(distt3)
    dist4.append(distt4)


# Setting up the LpProblem

model = pp.LpProblem (name='Well_Optimize', sense=pp.LpMinimize)

 
#Variables

P1 = pp.LpVariable("P1", lowBound=0, cat='Continuous') #Pumping from Well 1

P2 = pp.LpVariable("P2", lowBound=0, cat='Continuous') #Pumping from Well 2

P3 = pp.LpVariable("P3", lowBound=0, cat='Continuous') #Pumping from Well 3

P4 = pp.LpVariable("P4", lowBound=0, cat='Continuous') #Pumping from Well 4

P5 = pp.LpVariable("P5", lowBound=0, cat='Continuous') #Pumping from Well 5

P6 = pp.LpVariable("P6", lowBound=0, cat='Continuous') #Pumping from Well 6

P7 = pp.LpVariable("P7", lowBound=0, cat='Continuous') #Pumping from Well 7

P8 = pp.LpVariable("P8", lowBound=0, cat='Continuous') #Pumping from Well 8

P9 = pp.LpVariable("P9", lowBound=0, cat='Continuous') #Pumping from Well 9

P10 = pp.LpVariable("P10", lowBound=0, cat='Continuous') #Pumping from Well 10

 

 # Thesi values for 4 corners

r11 = pd.to_numeric(dist1)
r22 = pd.to_numeric(dist2)
r33 = pd.to_numeric(dist3)
r44 = pd.to_numeric(dist4)

theis1 = well_func(r11, S, K, t)

theis2 = well_func(r22, S, K, t)

theis3 = well_func(r33, S, K, t)

theis4 = well_func(r44, S, K, t)

#Terms for objective function

#Drawdown in corner 1

DD_1_1 = (P1 * theis1[0]) / (4*math.pi*K*dist1[0]) #Due to well 1

DD_1_2 = (P2 * theis1[1]) / (4*math.pi*K*dist1[1]) #Due to well 2

DD_1_3 = (P3 * theis1[2]) / (4*math.pi*K*dist1[2]) #Due to well 3

DD_1_4 = (P4 * theis1[3]) / (4*math.pi*K*dist1[3]) #Due to well 4

DD_1_5 = (P5 * theis1[4]) / (4*math.pi*K*dist1[4]) #Due to well 5

DD_1_6 = (P6 * theis1[5]) / (4*math.pi*K*dist1[5]) #Due to well 6

DD_1_7 = (P7 * theis1[6]) / (4*math.pi*K*dist1[6]) #Due to well 7

DD_1_8 = (P8 * theis1[7]) / (4*math.pi*K*dist1[7]) #Due to well 8

DD_1_9 = (P9 * theis1[8]) / (4*math.pi*K*dist1[8]) #Due to well 9

DD_1_10 = (P10 * theis1[9]) / (4*math.pi*K*dist1[9]) #Due to well 10

 

DD_1 = DD_1_1 + DD_1_2 + DD_1_3 + DD_1_4 + DD_1_5 + DD_1_6 + DD_1_7 + DD_1_8 + DD_1_9 + DD_1_10

#Drawdown in corner 2

DD_2_1 = (P1 * theis1[0]) / (4*math.pi*K*dist1[0]) #Due to well 1

DD_2_2 = (P2 * theis2[1]) / (4*math.pi*K*dist2[1]) #Due to well 2

DD_2_3 = (P3 * theis2[2]) / (4*math.pi*K*dist2[2]) #Due to well 3

DD_2_4 = (P4 * theis2[3]) / (4*math.pi*K*dist2[3]) #Due to well 4

DD_2_5 = (P5 * theis2[4]) / (4*math.pi*K*dist2[4]) #Due to well 5

DD_2_6 = (P6 * theis2[5]) / (4*math.pi*K*dist2[5]) #Due to well 6

DD_2_7 = (P7 * theis2[6]) / (4*math.pi*K*dist2[6]) #Due to well 7

DD_2_8 = (P8 * theis2[7]) / (4*math.pi*K*dist2[7]) #Due to well 8

DD_2_9 = (P9 * theis2[8]) / (4*math.pi*K*dist2[8]) #Due to well 9

DD_2_10 = (P10 * theis2[9]) / (4*math.pi*K*dist2[9]) #Due to well 10

 

DD_2 = DD_2_1 + DD_2_2 + DD_2_3 + DD_2_4 + DD_2_5 + DD_2_6 + DD_2_7 + DD_2_8 + DD_2_9 + DD_2_10

 

#Drawdown in corner 3

DD_3_1 = (P1 * theis3[0]) / (4*math.pi*K*dist3[0]) #Due to well 1

DD_3_2 = (P2 * theis3[1]) / (4*math.pi*K*dist3[1]) #Due to well 2

DD_3_3 = (P3 * theis3[2]) / (4*math.pi*K*dist3[2]) #Due to well 3

DD_3_4 = (P4 * theis3[3]) / (4*math.pi*K*dist3[3]) #Due to well 4

DD_3_5 = (P5 * theis3[4]) / (4*math.pi*K*dist3[4]) #Due to well 5

DD_3_6 = (P6 * theis3[5]) / (4*math.pi*K*dist3[5]) #Due to well 6

DD_3_7 = (P7 * theis3[6]) / (4*math.pi*K*dist3[6]) #Due to well 7

DD_3_8 = (P8 * theis3[7]) / (4*math.pi*K*dist3[7]) #Due to well 8

DD_3_9 = (P9 * theis3[8]) / (4*math.pi*K*dist3[8]) #Due to well 9

DD_3_10 = (P10 * theis1[9]) / (4*math.pi*K*dist1[9]) #Due to well 10

 

DD_3 = DD_3_1 + DD_3_2 + DD_3_3 + DD_3_4 + DD_3_5 + DD_3_6 + DD_3_7 + DD_3_8 + DD_3_9 + DD_3_10

 

 

#Drawdown in corner 4

DD_4_1 = (P1 * theis4[0]) / (4*math.pi*K*dist4[0]) #Due to well 1

DD_4_2 = (P2 * theis4[1]) / (4*math.pi*K*dist4[1]) #Due to well 2

DD_4_3 = (P3 * theis4[2]) / (4*math.pi*K*dist4[2]) #Due to well 3

DD_4_4 = (P4 * theis4[3]) / (4*math.pi*K*dist4[3]) #Due to well 4

DD_4_5 = (P5 * theis4[4]) / (4*math.pi*K*dist4[4]) #Due to well 5

DD_4_6 = (P6 * theis4[5]) / (4*math.pi*K*dist4[5]) #Due to well 6

DD_4_7 = (P7 * theis4[6]) / (4*math.pi*K*dist4[6]) #Due to well 7

DD_4_8 = (P8 * theis4[7]) / (4*math.pi*K*dist4[7]) #Due to well 8

DD_4_9 = (P9 * theis4[8]) / (4*math.pi*K*dist4[8]) #Due to well 9

DD_4_10 = (P10 * theis4[9]) / (4*math.pi*K*dist4[9]) #Due to well 10

 

DD_4 = DD_4_1 + DD_4_2 + DD_4_3 + DD_4_4 + DD_4_5 + DD_4_6 + DD_4_7 + DD_4_8 + DD_4_9 + DD_4_10

 

#Objective Function

model += DD_1 + DD_2 + DD_3 + DD_4
model.solve()