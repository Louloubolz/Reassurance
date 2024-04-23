#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:51:07 2022

@author: B216ZF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from mpl_toolkits.mplot3d import Axes3D

# Question 3

policies = pd.read_csv("/Users/B216ZF/Desktop/Cours de Réassurance/Cours/2023/TP/TP 4/Données/endorsement.csv", sep = ";", decimal = ",")
losses = pd.read_csv("/Users/B216ZF/Desktop/Cours de Réassurance/Cours/2023/TP/TP 4/Données/losses.csv", sep = ";", decimal = ",")
profile = pd.read_csv("/Users/B216ZF/Desktop/Cours de Réassurance/Cours/2023/TP/TP 4/Données/profile.csv", sep = ";", decimal = ",")

finalLosses = losses.groupby("UsualEndorsementId").agg({"LossTotal" : "sum"}).reset_index()
finalLosses = pd.merge(finalLosses, policies, on = "UsualEndorsementId", how = "outer")
finalLosses["DestructionRate"] = finalLosses["LossTotal"]/finalLosses["Exposure"]
finalLosses["DestructionRate"] = np.minimum(1, np.maximum(finalLosses["DestructionRate"], 0))

# Question 4

def G_emp(finalLosses, m):
    x = finalLosses["DestructionRate"]
    L_m = np.array([np.mean(np.minimum(x, i)) for i in m])
    L_1 = np.mean(x)
    return(L_m/L_1)

plt.plot(np.arange(0, 1.01, 0.01), G_emp(finalLosses, np.arange(0, 1.01, 0.01)))
plt.xlabel("Priorité")
plt.ylabel("Courbe d'exposition")

# Question 5

p = sum(finalLosses["DestructionRate"] == 1)/finalLosses.shape[0]
g = 1/p

esp = np.mean(finalLosses["DestructionRate"])

def esp_x(g, b):
    return((np.log(g*b)*(1-b))/(np.log(b)*(1-g*b)))

b = fsolve(lambda b: esp_x(g, b) - esp,
           0.1)

# Question 6

def G(m, g, b):
    return(np.log(((g-1)*b+(1-g*b)*b**m)/(1-b))/np.log(g*b))

def least_square(par, m):
    g = par[0]
    b = par[1]
    return(np.sqrt(np.sum((G(m, g, b) - G_emp(finalLosses, m))**2)))

par = minimize(lambda par: least_square(par, np.arange(0, 1.001, 0.001)),
               [34, 14],
               options = {"maxiter" : 500},
               method = "Nelder-Mead",
               bounds=((0,None), (0, None)))

plt.plot(np.arange(0, 1.01, 0.01), G_emp(finalLosses, np.arange(0, 1.01, 0.01)), "k--", label = "Courbe empirique")
plt.plot(np.arange(0, 1.01, 0.01), G(np.arange(0, 1.01, 0.01), par["x"][0], par["x"][1]), "r-", label = "Courbe moindres carrés")
plt.plot(np.arange(0, 1.01, 0.01), G(np.arange(0, 1.01, 0.01), g, b), "b-", label = "Courbe méthode des moments")
plt.xlabel("Priorité")
plt.ylabel("Courbe d'exposition")
plt.legend()

# Question 7

attachement = 1.5e6
limit = 3.5e6
lossRatio = 0.6

def price_unlimited_layer(attachement, profile, g, b, lossRatio):
    prf = profile.copy(deep = True)
    prf["MeanExpo"] = prf["ExposureAxaShare"]/prf["EarnedNbRisks"]
    prf["MeanRetention"] = np.minimum(1, attachement/prf["MeanExpo"])
    prf["ExposureCurve"] = G(prf["MeanRetention"], g, b)
    prf["MeanLoss"] = lossRatio*prf["EarnedPremium"]
    prf["ReinsurancePremiumAttachement"] = prf["MeanLoss"]*(1-prf["ExposureCurve"])
    return(prf)

def price_layer(attachement, limit, profile, g, b, lossRatio):
    price_at = price_unlimited_layer(attachement, profile, g, b, lossRatio)
    price_li = price_unlimited_layer(attachement+limit, profile, g, b, lossRatio)
    price = price_at.copy(deep=True)
    price["ReinsurancePremiumLimit"] = price_li["ReinsurancePremiumAttachement"]
    price["ReinsurancePremium"] = price["ReinsurancePremiumAttachement"] - price["ReinsurancePremiumLimit"]
    return(price[["MinExpo", "MaxExpo", "EarnedNbRisks", "EarnedPremium", "ExposureAxaShare", "ReinsurancePremiumAttachement", "ReinsurancePremiumLimit", "ReinsurancePremium"]])

rate = price_layer(attachement, limit, profile, par["x"][0], par["x"][1], lossRatio)
    
# Question 8

attachement = np.arange(5e6, 60e6, 10e6)
limit = np.arange(10e6, 110e6, 10e6)
z_expo = np.zeros((len(attachement), len(limit)))

for i in range(len(attachement)):
    for j in range(len(limit)):
        price_xs = price_layer(attachement[i], limit[j], profile, par["x"][0], par["x"][1], lossRatio)
        z_expo[i,j] = np.sum(price_xs["ReinsurancePremium"])

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection = "3d")
ax.plot_surface(np.outer(np.ones(len(limit)), attachement),
                np.outer(np.ones(len(attachement)), limit).transpose(),
                z_expo.transpose(),
                cmap = plt.cm.Spectral_r)
ax.set_xlabel("Priorité")
ax.set_ylabel("Portée")
ax.set_zlabel("Prix")

# Question 11

def G_p(m, g, b):
    return(((1-g*b)*np.log(b))/(np.log(g*b)*(1-b)))

profile["G_p"] = G_p(0, par["x"][0], par["x"][1])
profile["Lambda"] = profile["G_p"]*lossRatio*profile["EarnedPremium"]/(profile["ExposureAxaShare"]/profile["EarnedNbRisks"])
profile["Band"] = np.arange(profile.shape[0])


def simulations(n, g, b, profile, h, attachement, limit):
    n_loss = np.random.poisson(profile.loc[profile["Band"] == h, "Lambda"], n)
    loss = pd.DataFrame({"year" : np.repeat(np.arange(n), n_loss)})
    loss["r"] = np.random.uniform(0, 1, np.sum(n_loss))
    threshold = 1 - 1/g
    loss["total_loss"] = (loss["r"] > threshold)
    loss["loss"] = 1 - np.log(((1-b)/(1-loss["r"]) - (1 - g*b))/(g-1))/np.log(b)
    loss.loc[loss["total_loss"],:] = 1
    loss["loss_value"] = loss["loss"]*profile.loc[profile["Band"] == h, "ExposureAxaShare"].values/profile.loc[profile["Band"] == h, "EarnedNbRisks"].values
    loss["cession"] = np.minimum(np.maximum(loss["loss_value"] - attachement, 0), limit)
    return(loss)
    
n = 1000
attachement = 1.5e6
limit = 3.5e6
losses = simulations(n, par["x"][0], par["x"][1], profile, 0, attachement, limit)

for i in range(1, profile.shape[0]):
    losses = pd.concat([losses, simulations(n, par["x"][0], par["x"][1], profile, i, attachement, limit)])
    
rate = np.sum(losses.groupby("year").agg({"cession" : "sum"}))/n
    
