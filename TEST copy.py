import numpy as np
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import matplotlib.pyplot as plt
import json
import time
import TestExcel as te
import xlwt
from xlwt import Workbook
import pandas as pd


filename = "Medium Instances 1,5/Medium4,1.5.xlsx"
file_to_save = 'Results/Small Instances 1,5/Small_4,1.5.xls'
te.main(filename)

start_time = time.time()

passengers_json = json.load(open('sample_passenger.json'))
drivers_json = json.load(open('sample_driver.json'))


rnd = np.random
rnd.seed(0)

nr_passengers = len(passengers_json)
nr_drivers = len(drivers_json)

'''Coordinates '''
xc = []
yc = []

""""Delete this"""
def add_coordinates():
    """ Create coordinates (x, y) for the origins and destinations of drivers and passengers
        :return:
        """
    for drivers in drivers_json:
        xc.append(drivers_json[drivers]['origin_xc'])
        yc.append(drivers_json[drivers]['origin_yc'])
    for passengers in passengers_json:
        xc.append(passengers_json[passengers]['origin_xc'])
        yc.append(passengers_json[passengers]['origin_yc'])
    for passengers in passengers_json:
        xc.append(passengers_json[passengers]['destination_xc'])
        yc.append(passengers_json[passengers]['destination_yc'])
    for drivers in drivers_json:
        xc.append(drivers_json[drivers]['destination_xc'])
        yc.append(drivers_json[drivers]['destination_yc'])


add_coordinates()


'''Sets'''

D = [i for i in range(nr_drivers)]
N = [i for i in range(nr_passengers * 2 + nr_drivers * 2)]
NP = N[int(len(D)):int(len(N) / 2)]
ND = N[int(len(N) / 2):-int(len(D))]
A = [(i, j) for i in N for j in N if i != j]



delivery_and_pickup_node_pairs = {ND[i]: NP[i] for i in range(len(ND))}
pickup_and_delivery_node_pairs = {NP[i]: ND[i] for i in range(len(ND))}

'''Parameters'''
o_k = {}
d_k = {}
T_k = {}

"""Create T_ij"""
distance_matrix = {('Knappskog', 'Knappskog'): 0.1, ('Knappskog', 'Kolltveit'): 6, ('Knappskog', 'Kårtveit'): 8, ('Knappskog', 'Telavåg'): 28, ('Knappskog', 'Straume'): 9, ('Knappskog', 'Knarrevik'): 10, ('Knappskog', 'Bildøyna'): 7, ('Knappskog', 'Spjeld'): 2, ('Knappskog', 'Foldnes'): 13, ('Knappskog', 'Brattholmen'): 12, ('Knappskog', 'Blomøy'): 22, ('Knappskog', 'Hellesøy'): 41, ('Knappskog', 'Arefjord'): 12, ('Knappskog', 'Ebbesvik'): 15, ('Knappskog', 'Landro'): 10, ('Knappskog', 'Hjelteryggen'): 13, ('Knappskog', 'Skogsvåg'): 20, ('Knappskog', 'Kleppestø'): 21, ('Knappskog', 'Solsvik'): 12, ('Knappskog', 'Rongøy'): 17, ('Knappskog', 'Hammarsland'): 19, ('Knappskog', 'Træsneset'): 23, ('Knappskog', 'Tofterøy'): 31, ('Knappskog', 'Bergenhus'): 27, ('Knappskog', 'Laksevåg'): 20, ('Knappskog', 'Ytrebygda'): 25, ('Knappskog', 'Årstad'): 27, ('Kolltveit', 'Knappskog'): 6, ('Kolltveit', 'Kolltveit'): 0.1, ('Kolltveit', 'Kårtveit'): 13, ('Kolltveit', 'Telavåg'): 26, ('Kolltveit', 'Straume'): 7, ('Kolltveit', 'Knarrevik'): 7, ('Kolltveit', 'Bildøyna'): 5, ('Kolltveit', 'Spjeld'): 7, ('Kolltveit', 'Foldnes'): 11, ('Kolltveit', 'Brattholmen'): 10, ('Kolltveit', 'Blomøy'): 27, ('Kolltveit', 'Hellesøy'): 46, ('Kolltveit', 'Arefjord'): 10, ('Kolltveit', 'Ebbesvik'): 13, ('Kolltveit', 'Landro'): 15, ('Kolltveit', 'Hjelteryggen'): 10, ('Kolltveit', 'Skogsvåg'): 18, ('Kolltveit', 'Kleppestø'): 18, ('Kolltveit', 'Solsvik'): 17, ('Kolltveit', 'Rongøy'): 22, ('Kolltveit', 'Hammarsland'): 16, ('Kolltveit', 'Træsneset'): 21, ('Kolltveit', 'Tofterøy'): 29, ('Kolltveit', 'Bergenhus'): 25, ('Kolltveit', 'Laksevåg'): 18, ('Kolltveit', 'Ytrebygda'): 23, ('Kolltveit', 'Årstad'): 25, ('Kårtveit', 'Knappskog'): 8, ('Kårtveit', 'Kolltveit'): 13, ('Kårtveit', 'Kårtveit'): 0.1, ('Kårtveit', 'Telavåg'): 36, ('Kårtveit', 'Straume'): 17, ('Kårtveit', 'Knarrevik'): 17, ('Kårtveit', 'Bildøyna'): 15, ('Kårtveit', 'Spjeld'): 7, ('Kårtveit', 'Foldnes'): 21, ('Kårtveit', 'Brattholmen'): 20, ('Kårtveit', 'Blomøy'): 20, ('Kårtveit', 'Hellesøy'): 39, ('Kårtveit', 'Arefjord'): 20, ('Kårtveit', 'Ebbesvik'): 23, ('Kårtveit', 'Landro'): 7, ('Kårtveit', 'Hjelteryggen'): 20, ('Kårtveit', 'Skogsvåg'): 28, ('Kårtveit', 'Kleppestø'): 29, ('Kårtveit', 'Solsvik'): 10, ('Kårtveit', 'Rongøy'): 15, ('Kårtveit', 'Hammarsland'): 26, ('Kårtveit', 'Træsneset'): 31, ('Kårtveit', 'Tofterøy'): 39, ('Kårtveit', 'Bergenhus'): 35, ('Kårtveit', 'Laksevåg'): 28, ('Kårtveit', 'Ytrebygda'): 33, ('Kårtveit', 'Årstad'): 35, ('Telavåg', 'Knappskog'): 28, ('Telavåg', 'Kolltveit'): 26, ('Telavåg', 'Kårtveit'): 35, ('Telavåg', 'Telavåg'): 0.1, ('Telavåg', 'Straume'): 29, ('Telavåg', 'Knarrevik'): 29, ('Telavåg', 'Bildøyna'): 27, ('Telavåg', 'Spjeld'): 30, ('Telavåg', 'Foldnes'): 33, ('Telavåg', 'Brattholmen'): 32, ('Telavåg', 'Blomøy'): 50, ('Telavåg', 'Hellesøy'): 69, ('Telavåg', 'Arefjord'): 32, ('Telavåg', 'Ebbesvik'): 35, ('Telavåg', 'Landro'): 37, ('Telavåg', 'Hjelteryggen'): 32, ('Telavåg', 'Skogsvåg'): 14, ('Telavåg', 'Kleppestø'): 40, ('Telavåg', 'Solsvik'): 39, ('Telavåg', 'Rongøy'): 44, ('Telavåg', 'Hammarsland'): 10, ('Telavåg', 'Træsneset'): 10, ('Telavåg', 'Tofterøy'): 19, ('Telavåg', 'Bergenhus'): 47, ('Telavåg', 'Laksevåg'): 40, ('Telavåg', 'Ytrebygda'): 45, ('Telavåg', 'Årstad'): 47, ('Straume', 'Knappskog'): 9, ('Straume', 'Kolltveit'): 7, ('Straume', 'Kårtveit'): 16, ('Straume', 'Telavåg'): 29, ('Straume', 'Straume'): 0.1 ,('Straume', 'Knarrevik'): 4, ('Straume', 'Bildøyna'): 4, ('Straume', 'Spjeld'): 10, ('Straume', 'Foldnes'): 6, ('Straume', 'Brattholmen'): 5, ('Straume', 'Blomøy'): 30, ('Straume', 'Hellesøy'): 49, ('Straume', 'Straume'): 0, ('Straume', 'Arefjord'): 5, ('Straume', 'Ebbesvik'): 7, ('Straume', 'Landro'): 18, ('Straume', 'Hjelteryggen'): 7, ('Straume', 'Skogsvåg'): 21, ('Straume', 'Kleppestø'): 15, ('Straume', 'Solsvik'): 20, ('Straume', 'Rongøy'): 25, ('Straume', 'Hammarsland'): 19, ('Straume', 'Træsneset'): 24, ('Straume', 'Tofterøy'): 32, ('Straume', 'Bergenhus'): 22, ('Straume', 'Laksevåg'): 15, ('Straume', 'Ytrebygda'): 20, ('Straume', 'Årstad'): 22, ('Knarrevik', 'Knappskog'): 9, ('Knarrevik', 'Kolltveit'): 7, ('Knarrevik', 'Kårtveit'): 17, ('Knarrevik', 'Telavåg'): 29, ('Knarrevik', 'Straume'): 4, ('Knarrevik', 'Knarrevik'): 0.1, ('Knarrevik', 'Bildøyna'): 4, ('Knarrevik', 'Spjeld'): 11, ('Knarrevik', 'Foldnes'): 7, ('Knarrevik', 'Brattholmen'): 5, ('Knarrevik', 'Blomøy'): 31, ('Knarrevik', 'Hellesøy'): 50, ('Knarrevik', 'Arefjord'): 4, ('Knarrevik', 'Ebbesvik'): 9, ('Knarrevik', 'Landro'): 19, ('Knarrevik', 'Hjelteryggen'): 3, ('Knarrevik', 'Skogsvåg'): 21, ('Knarrevik', 'Kleppestø'): 12, ('Knarrevik', 'Solsvik'): 20, ('Knarrevik', 'Rongøy'): 26, ('Knarrevik', 'Hammarsland'): 19, ('Knarrevik', 'Træsneset'): 24, ('Knarrevik', 'Tofterøy'): 32, ('Knarrevik', 'Bergenhus'): 18, ('Knarrevik', 'Laksevåg'): 11, ('Knarrevik', 'Ytrebygda'): 16, ('Knarrevik', 'Årstad'): 18, ('Bildøyna', 'Knappskog'): 8, ('Bildøyna', 'Kolltveit'): 5, ('Bildøyna', 'Kårtveit'): 15, ('Bildøyna', 'Telavåg'): 27, ('Bildøyna', 'Straume'): 4, ('Bildøyna', 'Knarrevik'): 4, ('Bildøyna', 'Spjeld'): 9, ('Bildøyna', 'Bildøyna'): 0.1, ('Bildøyna', 'Foldnes'): 8, ('Bildøyna', 'Brattholmen'): 7, ('Bildøyna', 'Blomøy'): 29, ('Bildøyna', 'Hellesøy'): 48, ('Bildøyna', 'Arefjord'): 7, ('Bildøyna', 'Ebbesvik'): 10, ('Bildøyna', 'Landro'): 17, ('Bildøyna', 'Hjelteryggen'): 7, ('Bildøyna', 'Skogsvåg'): 19, ('Bildøyna', 'Kleppestø'): 16, ('Bildøyna', 'Solsvik'): 19, ('Bildøyna', 'Rongøy'): 24, ('Bildøyna', 'Hammarsland'): 18, ('Bildøyna', 'Træsneset'): 22, ('Bildøyna', 'Tofterøy'): 31, ('Bildøyna', 'Bergenhus'): 22, ('Bildøyna', 'Laksevåg'): 15, ('Bildøyna', 'Ytrebygda'): 20, ('Bildøyna', 'Årstad'): 22, ('Spjeld', 'Knappskog'): 2, ('Spjeld', 'Kolltveit'): 7, ('Spjeld', 'Kårtveit'): 6, ('Spjeld', 'Telavåg'): 30, ('Spjeld', 'Straume'): 11, ('Spjeld', 'Knarrevik'): 11, ('Spjeld', 'Bildøyna'): 9, ('Spjeld', 'Spjeld'): 0.1, ('Spjeld', 'Foldnes'): 15, ('Spjeld', 'Brattholmen'): 14, ('Spjeld', 'Blomøy'): 21, ('Spjeld', 'Hellesøy'): 40, ('Spjeld', 'Arefjord'): 14, ('Spjeld', 'Ebbesvik'): 17, ('Spjeld', 'Landro'): 8, ('Spjeld', 'Hjelteryggen'): 14, ('Spjeld', 'Skogsvåg'): 22, ('Spjeld', 'Kleppestø'): 22, ('Spjeld', 'Solsvik'): 10, ('Spjeld', 'Rongøy'): 15, ('Spjeld', 'Hammarsland'): 20, ('Spjeld', 'Træsneset'): 25, ('Spjeld', 'Tofterøy'): 33, ('Spjeld', 'Bergenhus'): 29, ('Spjeld', 'Laksevåg'): 22, ('Spjeld', 'Ytrebygda'): 27, ('Spjeld', 'Årstad'): 29, ('Foldnes', 'Knappskog'): 12, ('Foldnes', 'Kolltveit'): 10, ('Foldnes', 'Kårtveit'): 20, ('Foldnes', 'Telavåg'): 32, ('Foldnes', 'Straume'): 6, ('Foldnes', 'Knarrevik'): 8, ('Foldnes', 'Bildøyna'): 7, ('Foldnes', 'Spjeld'): 14, ('Foldnes', 'Foldnes'): 0.1, ('Foldnes', 'Brattholmen'): 10, ('Foldnes', 'Blomøy'): 34, ('Foldnes', 'Hellesøy'): 53, ('Foldnes', 'Arefjord'): 10, ('Foldnes', 'Ebbesvik'): 13, ('Foldnes', 'Landro'): 21, ('Foldnes', 'Hjelteryggen'): 11, ('Foldnes', 'Skogsvåg'): 24, ('Foldnes', 'Kleppestø'): 19, ('Foldnes', 'Solsvik'): 23, ('Foldnes', 'Rongøy'): 28, ('Foldnes', 'Hammarsland'): 22, ('Foldnes', 'Træsneset'): 27, ('Foldnes', 'Tofterøy'): 35, ('Foldnes', 'Bergenhus'): 26, ('Foldnes', 'Laksevåg'): 19, ('Foldnes', 'Ytrebygda'): 24, ('Foldnes', 'Årstad'): 26, ('Brattholmen', 'Knappskog'): 13, ('Brattholmen', 'Kolltveit'): 11, ('Brattholmen', 'Kårtveit'): 20, ('Brattholmen', 'Telavåg'): 33, ('Brattholmen', 'Straume'): 5, ('Brattholmen', 'Knarrevik'): 6, ('Brattholmen', 'Bildøyna'): 8, ('Brattholmen', 'Spjeld'): 14, ('Brattholmen', 'Foldnes'): 10,('Brattholmen', 'Brattholmen'): 0.1,  ('Brattholmen', 'Blomøy'): 34, ('Brattholmen', 'Hellesøy'): 53, ('Brattholmen', 'Arefjord'): 4, ('Brattholmen', 'Ebbesvik'): 5, ('Brattholmen', 'Landro'): 22, ('Brattholmen', 'Hjelteryggen'): 9, ('Brattholmen', 'Skogsvåg'): 25, ('Brattholmen', 'Kleppestø'): 17, ('Brattholmen', 'Solsvik'): 24, ('Brattholmen', 'Rongøy'): 29, ('Brattholmen', 'Hammarsland'): 23, ('Brattholmen', 'Træsneset'): 28, ('Brattholmen', 'Tofterøy'): 36, ('Brattholmen', 'Bergenhus'): 23, ('Brattholmen', 'Laksevåg'): 16, ('Brattholmen', 'Ytrebygda'): 21, ('Brattholmen', 'Årstad'): 23, ('Blomøy', 'Knappskog'): 23, ('Blomøy', 'Kolltveit'): 28, ('Blomøy', 'Kårtveit'): 21, ('Blomøy', 'Telavåg'): 51, ('Blomøy', 'Straume'): 32, ('Blomøy', 'Knarrevik'): 32, ('Blomøy', 'Bildøyna'): 30, ('Blomøy', 'Spjeld'): 22, ('Blomøy', 'Foldnes'): 36, ('Blomøy', 'Brattholmen'): 35,('Blomøy', 'Blomøy'): 0.1,  ('Blomøy', 'Hellesøy'): 23, ('Blomøy', 'Arefjord'): 35, ('Blomøy', 'Ebbesvik'): 38, ('Blomøy', 'Landro'): 20, ('Blomøy', 'Hjelteryggen'): 35, ('Blomøy', 'Skogsvåg'): 43, ('Blomøy', 'Kleppestø'): 43, ('Blomøy', 'Solsvik'): 18, ('Blomøy', 'Rongøy'): 7, ('Blomøy', 'Hammarsland'): 41, ('Blomøy', 'Træsneset'): 46, ('Blomøy', 'Tofterøy'): 54, ('Blomøy', 'Bergenhus'): 50, ('Blomøy', 'Laksevåg'): 43, ('Blomøy', 'Ytrebygda'): 48, ('Blomøy', 'Årstad'): 50, ('Hellesøy', 'Knappskog'): 41, ('Hellesøy', 'Kolltveit'): 46, ('Hellesøy', 'Kårtveit'): 39, ('Hellesøy', 'Telavåg'): 69, ('Hellesøy', 'Straume'): 50, ('Hellesøy', 'Knarrevik'): 50, ('Hellesøy', 'Bildøyna'): 48, ('Hellesøy', 'Spjeld'): 40, ('Hellesøy', 'Foldnes'): 54, ('Hellesøy', 'Brattholmen'): 53, ('Hellesøy', 'Blomøy'): 24, ('Hellesøy', 'Hellesøy'): 0.1, ('Hellesøy', 'Arefjord'): 53, ('Hellesøy', 'Ebbesvik'): 56, ('Hellesøy', 'Landro'): 38, ('Hellesøy', 'Hjelteryggen'): 53, ('Hellesøy', 'Skogsvåg'): 61, ('Hellesøy', 'Kleppestø'): 61, ('Hellesøy', 'Solsvik'): 36, ('Hellesøy', 'Rongøy'): 25, ('Hellesøy', 'Hammarsland'): 59, ('Hellesøy', 'Træsneset'): 64, ('Hellesøy', 'Tofterøy'): 72, ('Hellesøy', 'Bergenhus'): 68, ('Hellesøy', 'Laksevåg'): 61, ('Hellesøy', 'Ytrebygda'): 66, ('Hellesøy', 'Årstad'): 68, ('Arefjord', 'Knappskog'): 13, ('Arefjord', 'Kolltveit'): 11, ('Arefjord', 'Kårtveit'): 20, ('Arefjord', 'Telavåg'): 33, ('Arefjord', 'Straume'): 5, ('Arefjord', 'Knarrevik'): 4, ('Arefjord', 'Bildøyna'): 8, ('Arefjord', 'Spjeld'): 15, ('Arefjord', 'Foldnes'): 10, ('Arefjord', 'Brattholmen'): 4, ('Arefjord', 'Blomøy'): 34, ('Arefjord', 'Hellesøy'): 53, ('Arefjord', 'Arefjord'): 0.1, ('Arefjord', 'Ebbesvik'): 7, ('Arefjord', 'Landro'): 22, ('Arefjord', 'Hjelteryggen'): 7, ('Arefjord', 'Skogsvåg'): 25, ('Arefjord', 'Kleppestø'): 16, ('Arefjord', 'Solsvik'): 24, ('Arefjord', 'Rongøy'): 29, ('Arefjord', 'Hammarsland'): 23, ('Arefjord', 'Træsneset'): 28, ('Arefjord', 'Tofterøy'): 36, ('Arefjord', 'Bergenhus'): 22, ('Arefjord', 'Laksevåg'): 15, ('Arefjord', 'Ytrebygda'): 20, ('Arefjord', 'Årstad'): 22, ('Ebbesvik', 'Knappskog'): 16, ('Ebbesvik', 'Kolltveit'): 14, ('Ebbesvik', 'Kårtveit'): 23, ('Ebbesvik', 'Telavåg'): 35, ('Ebbesvik', 'Straume'): 7, ('Ebbesvik', 'Knarrevik'): 8, ('Ebbesvik', 'Bildøyna'): 10, ('Ebbesvik', 'Spjeld'): 17, ('Ebbesvik', 'Foldnes'): 13, ('Ebbesvik', 'Brattholmen'): 5, ('Ebbesvik', 'Blomøy'): 37, ('Ebbesvik', 'Hellesøy'): 56, ('Ebbesvik', 'Arefjord'): 7, ('Ebbesvik', 'Arefjord'): 0.1, ('Ebbesvik', 'Landro'): 25, ('Ebbesvik', 'Hjelteryggen'): 11, ('Ebbesvik', 'Skogsvåg'): 27, ('Ebbesvik', 'Kleppestø'): 19, ('Ebbesvik', 'Solsvik'): 27, ('Ebbesvik', 'Rongøy'): 32, ('Ebbesvik', 'Hammarsland'): 25, ('Ebbesvik', 'Træsneset'): 30, ('Ebbesvik', 'Tofterøy'): 38, ('Ebbesvik', 'Bergenhus'): 26, ('Ebbesvik', 'Laksevåg'): 19, ('Ebbesvik', 'Ytrebygda'): 24, ('Ebbesvik', 'Årstad'): 26, ('Landro', 'Knappskog'): 10, ('Landro', 'Kolltveit'): 15, ('Landro', 'Kårtveit'): 7, ('Landro', 'Telavåg'): 37, ('Landro', 'Straume'): 18, ('Landro', 'Knarrevik'): 19, ('Landro', 'Bildøyna'): 16, ('Landro', 'Spjeld'): 8, ('Landro', 'Foldnes'): 22, ('Landro', 'Brattholmen'): 22, ('Landro', 'Blomøy'): 19, ('Landro', 'Hellesøy'): 38, ('Landro', 'Arefjord'): 22, ('Landro', 'Ebbesvik'): 24, ('Landro', 'Landro'): 0.1, ('Landro', 'Hjelteryggen'): 22, ('Landro', 'Skogsvåg'): 29, ('Landro', 'Kleppestø'): 30, ('Landro', 'Solsvik'): 3, ('Landro', 'Rongøy'): 13, ('Landro', 'Hammarsland'): 28, ('Landro', 'Træsneset'): 32, ('Landro', 'Tofterøy'): 41, ('Landro', 'Bergenhus'): 36, ('Landro', 'Laksevåg'): 29, ('Landro', 'Ytrebygda'): 34, ('Landro', 'Årstad'): 37, ('Hjelteryggen', 'Knappskog'): 13, ('Hjelteryggen', 'Kolltveit'): 10, ('Hjelteryggen', 'Kårtveit'): 20, ('Hjelteryggen', 'Telavåg'): 32, ('Hjelteryggen', 'Straume'): 7, ('Hjelteryggen', 'Knarrevik'): 4, ('Hjelteryggen', 'Bildøyna'): 7, ('Hjelteryggen', 'Spjeld'): 14, ('Hjelteryggen', 'Foldnes'): 10, ('Hjelteryggen', 'Brattholmen'): 9, ('Hjelteryggen', 'Blomøy'): 34, ('Hjelteryggen', 'Hellesøy'): 53, ('Hjelteryggen', 'Arefjord'): 7, ('Hjelteryggen', 'Ebbesvik'): 12, ('Hjelteryggen', 'Landro'): 22, ('Hjelteryggen', 'Hjelteryggen'): 22, ('Hjelteryggen', 'Skogsvåg'): 24, ('Hjelteryggen', 'Kleppestø'): 14, ('Hjelteryggen', 'Solsvik'): 24, ('Hjelteryggen', 'Rongøy'): 29, ('Hjelteryggen', 'Hammarsland'): 23, ('Hjelteryggen', 'Træsneset'): 27, ('Hjelteryggen', 'Tofterøy'): 36, ('Hjelteryggen', 'Bergenhus'): 21, ('Hjelteryggen', 'Laksevåg'): 14, ('Hjelteryggen', 'Ytrebygda'): 19, ('Hjelteryggen', 'Årstad'): 21, ('Skogsvåg', 'Knappskog'): 20, ('Skogsvåg', 'Kolltveit'): 18, ('Skogsvåg', 'Kårtveit'): 28, ('Skogsvåg', 'Telavåg'): 15, ('Skogsvåg', 'Straume'): 21, ('Skogsvåg', 'Knarrevik'): 21, ('Skogsvåg', 'Bildøyna'): 19, ('Skogsvåg', 'Spjeld'): 22, ('Skogsvåg', 'Foldnes'): 25, ('Skogsvåg', 'Brattholmen'): 24, ('Skogsvåg', 'Blomøy'): 42, ('Skogsvåg', 'Hellesøy'): 61, ('Skogsvåg', 'Arefjord'): 24, ('Skogsvåg', 'Ebbesvik'): 27, ('Skogsvåg', 'Landro'): 29, ('Skogsvåg', 'Hjelteryggen'): 24, ('Skogsvåg', 'Skogsvåg'): 0.1, ('Skogsvåg', 'Kleppestø'): 32, ('Skogsvåg', 'Solsvik'): 31, ('Skogsvåg', 'Rongøy'): 36, ('Skogsvåg', 'Hammarsland'): 5, ('Skogsvåg', 'Træsneset'): 10, ('Skogsvåg', 'Tofterøy'): 18, ('Skogsvåg', 'Bergenhus'): 39, ('Skogsvåg', 'Laksevåg'): 32, ('Skogsvåg', 'Ytrebygda'): 37, ('Skogsvåg', 'Årstad'): 39, ('Kleppestø', 'Knappskog'): 21, ('Kleppestø', 'Kolltveit'): 18, ('Kleppestø', 'Kårtveit'): 28, ('Kleppestø', 'Telavåg'): 40, ('Kleppestø', 'Straume'): 15, ('Kleppestø', 'Knarrevik'): 12, ('Kleppestø', 'Bildøyna'): 15, ('Kleppestø', 'Spjeld'): 22, ('Kleppestø', 'Foldnes'): 18, ('Kleppestø', 'Brattholmen'): 17, ('Kleppestø', 'Blomøy'): 42, ('Kleppestø', 'Hellesøy'): 61, ('Kleppestø', 'Arefjord'): 15, ('Kleppestø', 'Ebbesvik'): 20, ('Kleppestø', 'Landro'): 30, ('Kleppestø', 'Hjelteryggen'): 14, ('Kleppestø', 'Skogsvåg'): 32, ('Kleppestø', 'Kleppestø'): 0.1,('Kleppestø', 'Solsvik'): 32, ('Kleppestø', 'Rongøy'): 37, ('Kleppestø', 'Hammarsland'): 31, ('Kleppestø', 'Træsneset'): 35, ('Kleppestø', 'Tofterøy'): 44, ('Kleppestø', 'Bergenhus'): 22, ('Kleppestø', 'Laksevåg'): 15, ('Kleppestø', 'Ytrebygda'): 20, ('Kleppestø', 'Årstad'): 22, ('Solsvik', 'Knappskog'): 12, ('Solsvik', 'Kolltveit'): 17, ('Solsvik', 'Kårtveit'): 9, ('Solsvik', 'Telavåg'): 39, ('Solsvik', 'Straume'): 20, ('Solsvik', 'Knarrevik'): 21, ('Solsvik', 'Bildøyna'): 18, ('Solsvik', 'Spjeld'): 10, ('Solsvik', 'Foldnes'): 24, ('Solsvik', 'Brattholmen'): 23, ('Solsvik', 'Blomøy'): 16, ('Solsvik', 'Hellesøy'): 35, ('Solsvik', 'Arefjord'): 24, ('Solsvik', 'Ebbesvik'): 26, ('Solsvik', 'Landro'): 4, ('Solsvik', 'Hjelteryggen'): 24, ('Solsvik', 'Skogsvåg'): 31, ('Solsvik', 'Kleppestø'): 32,('Solsvik', 'Solsvik'): 0.1, ('Solsvik', 'Rongøy'): 11, ('Solsvik', 'Hammarsland'): 30, ('Solsvik', 'Træsneset'): 34, ('Solsvik', 'Tofterøy'): 43, ('Solsvik', 'Bergenhus'): 38, ('Solsvik', 'Laksevåg'): 31, ('Solsvik', 'Ytrebygda'): 36, ('Solsvik', 'Årstad'): 38, ('Rongøy', 'Knappskog'): 17, ('Rongøy', 'Kolltveit'): 22, ('Rongøy', 'Kårtveit'): 15, ('Rongøy', 'Telavåg'): 45, ('Rongøy', 'Straume'): 25, ('Rongøy', 'Knarrevik'): 26, ('Rongøy', 'Bildøyna'): 24, ('Rongøy', 'Spjeld'): 15, ('Rongøy', 'Foldnes'): 29, ('Rongøy', 'Brattholmen'): 29, ('Rongøy', 'Blomøy'): 6, ('Rongøy', 'Hellesøy'): 25, ('Rongøy', 'Arefjord'): 29, ('Rongøy', 'Ebbesvik'): 31, ('Rongøy', 'Landro'): 14, ('Rongøy', 'Hjelteryggen'): 29, ('Rongøy', 'Skogsvåg'): 37, ('Rongøy', 'Kleppestø'): 37, ('Rongøy', 'Solsvik'): 11,  ('Rongøy', 'Rongøy'): 11, ('Rongøy', 'Hammarsland'): 35, ('Rongøy', 'Træsneset'): 40, ('Rongøy', 'Tofterøy'): 48, ('Rongøy', 'Bergenhus'): 44, ('Rongøy', 'Laksevåg'): 37, ('Rongøy', 'Ytrebygda'): 42, ('Rongøy', 'Årstad'): 44, ('Hammarsland', 'Knappskog'): 19, ('Hammarsland', 'Kolltveit'): 16, ('Hammarsland', 'Kårtveit'): 26, ('Hammarsland', 'Telavåg'): 10, ('Hammarsland', 'Straume'): 19, ('Hammarsland', 'Knarrevik'): 19, ('Hammarsland', 'Bildøyna'): 17, ('Hammarsland', 'Spjeld'): 20, ('Hammarsland', 'Foldnes'): 23, ('Hammarsland', 'Brattholmen'): 22, ('Hammarsland', 'Blomøy'): 40, ('Hammarsland', 'Hellesøy'): 59, ('Hammarsland', 'Arefjord'): 22, ('Hammarsland', 'Ebbesvik'): 25, ('Hammarsland', 'Landro'): 28, ('Hammarsland', 'Hjelteryggen'): 22, ('Hammarsland', 'Skogsvåg'): 5, ('Hammarsland', 'Kleppestø'): 31, ('Hammarsland', 'Solsvik'): 30, ('Hammarsland', 'Rongøy'): 35, ('Hammarsland', 'Hammarsland'): 0.1,('Hammarsland', 'Træsneset'): 5, ('Hammarsland', 'Tofterøy'): 14, ('Hammarsland', 'Bergenhus'): 37, ('Hammarsland', 'Laksevåg'): 30, ('Hammarsland', 'Ytrebygda'): 35, ('Hammarsland', 'Årstad'): 37, ('Træsneset', 'Knappskog'): 24, ('Træsneset', 'Kolltveit'): 22, ('Træsneset', 'Kårtveit'): 32, ('Træsneset', 'Telavåg'): 12, ('Træsneset', 'Straume'): 25, ('Træsneset', 'Knarrevik'): 25, ('Træsneset', 'Bildøyna'): 23, ('Træsneset', 'Spjeld'): 26, ('Træsneset', 'Foldnes'): 29, ('Træsneset', 'Brattholmen'): 28, ('Træsneset', 'Blomøy'): 46, ('Træsneset', 'Hellesøy'): 65, ('Træsneset', 'Arefjord'): 28, ('Træsneset', 'Ebbesvik'): 31, ('Træsneset', 'Landro'): 34, ('Træsneset', 'Hjelteryggen'): 28, ('Træsneset', 'Skogsvåg'): 11, ('Træsneset', 'Kleppestø'): 37, ('Træsneset', 'Solsvik'): 36, ('Træsneset', 'Rongøy'): 41, ('Træsneset', 'Hammarsland'): 7,  ('Træsneset', 'Træsneset'): 0.1,('Træsneset', 'Tofterøy'): 10, ('Træsneset', 'Bergenhus'): 43, ('Træsneset', 'Laksevåg'): 36, ('Træsneset', 'Ytrebygda'): 41, ('Træsneset', 'Årstad'): 43, ('Tofterøy', 'Knappskog'): 31, ('Tofterøy', 'Kolltveit'): 29, ('Tofterøy', 'Kårtveit'): 39, ('Tofterøy', 'Telavåg'): 19, ('Tofterøy', 'Straume'): 32, ('Tofterøy', 'Knarrevik'): 32, ('Tofterøy', 'Bildøyna'): 30, ('Tofterøy', 'Spjeld'): 33, ('Tofterøy', 'Foldnes'): 36, ('Tofterøy', 'Brattholmen'): 35, ('Tofterøy', 'Blomøy'): 53, ('Tofterøy', 'Hellesøy'): 72, ('Tofterøy', 'Arefjord'): 35, ('Tofterøy', 'Ebbesvik'): 38, ('Tofterøy', 'Landro'): 41, ('Tofterøy', 'Hjelteryggen'): 35, ('Tofterøy', 'Skogsvåg'): 18, ('Tofterøy', 'Kleppestø'): 44, ('Tofterøy', 'Solsvik'): 42, ('Tofterøy', 'Rongøy'): 48, ('Tofterøy', 'Hammarsland'): 14, ('Tofterøy', 'Træsneset'): 10, ('Tofterøy', 'Tofterøy'): 0.1, ('Tofterøy', 'Bergenhus'): 50, ('Tofterøy', 'Laksevåg'): 43, ('Tofterøy', 'Ytrebygda'): 48, ('Tofterøy', 'Årstad'): 50, ('Bergenhus', 'Knappskog'): 27, ('Bergenhus', 'Kolltveit'): 25, ('Bergenhus', 'Kårtveit'): 35, ('Bergenhus', 'Telavåg'): 47, ('Bergenhus', 'Straume'): 22, ('Bergenhus', 'Knarrevik'): 18, ('Bergenhus', 'Bildøyna'): 22, ('Bergenhus', 'Spjeld'): 29, ('Bergenhus', 'Foldnes'): 26, ('Bergenhus', 'Brattholmen'): 23, ('Bergenhus', 'Blomøy'): 50, ('Bergenhus', 'Hellesøy'): 68, ('Bergenhus', 'Arefjord'): 22, ('Bergenhus', 'Ebbesvik'): 26, ('Bergenhus', 'Landro'): 36, ('Bergenhus', 'Hjelteryggen'): 21, ('Bergenhus', 'Skogsvåg'): 39, ('Bergenhus', 'Kleppestø'): 22, ('Bergenhus', 'Solsvik'): 38, ('Bergenhus', 'Rongøy'): 44, ('Bergenhus', 'Hammarsland'): 37, ('Bergenhus', 'Træsneset'): 43, ('Bergenhus', 'Bergenhus'): 0.1, ('Bergenhus', 'Laksevåg'): 14, ('Bergenhus', 'Ytrebygda'): 21, ('Bergenhus', 'Årstad'): 15, ('Laksevåg', 'Knappskog'): 20, ('Laksevåg', 'Kolltveit'): 18, ('Laksevåg', 'Kårtveit'): 28, ('Laksevåg', 'Telavåg'): 40, ('Laksevåg', 'Straume'): 15, ('Laksevåg', 'Knarrevik'): 11, ('Laksevåg', 'Bildøyna'): 15, ('Laksevåg', 'Spjeld'): 22, ('Laksevåg', 'Foldnes'): 19, ('Laksevåg', 'Brattholmen'): 16, ('Laksevåg', 'Blomøy'): 43, ('Laksevåg', 'Hellesøy'): 61, ('Laksevåg', 'Arefjord'): 15, ('Laksevåg', 'Ebbesvik'): 19, ('Laksevåg', 'Landro'): 29, ('Laksevåg', 'Hjelteryggen'): 14, ('Laksevåg', 'Skogsvåg'): 32, ('Laksevåg', 'Kleppestø'): 15, ('Laksevåg', 'Solsvik'): 31, ('Laksevåg', 'Rongøy'): 37, ('Laksevåg', 'Hammarsland'): 30, ('Laksevåg', 'Træsneset'): 36, ('Laksevåg', 'Bergenhus'): 15, ('Laksevåg', 'Ytrebygda'): 14,  ('Laksevåg', 'Laksevåg'): 0.1, ('Laksevåg', 'Årstad'): 15, ('Ytrebygda', 'Knappskog'): 25, ('Ytrebygda', 'Kolltveit'): 23, ('Ytrebygda', 'Kårtveit'): 33, ('Ytrebygda', 'Telavåg'): 45, ('Ytrebygda', 'Straume'): 20, ('Ytrebygda', 'Knarrevik'): 16, ('Ytrebygda', 'Bildøyna'): 20, ('Ytrebygda', 'Spjeld'): 27, ('Ytrebygda', 'Foldnes'): 24, ('Ytrebygda', 'Brattholmen'): 21, ('Ytrebygda', 'Blomøy'): 48, ('Ytrebygda', 'Hellesøy'): 66, ('Ytrebygda', 'Arefjord'): 20, ('Ytrebygda', 'Ebbesvik'): 24, ('Ytrebygda', 'Landro'): 34, ('Ytrebygda', 'Hjelteryggen'): 19, ('Ytrebygda', 'Skogsvåg'): 37, ('Ytrebygda', 'Kleppestø'): 20, ('Ytrebygda', 'Solsvik'): 36, ('Ytrebygda', 'Rongøy'): 42, ('Ytrebygda', 'Hammarsland'): 35, ('Ytrebygda', 'Træsneset'): 41, ('Ytrebygda', 'Bergenhus'): 21, ('Ytrebygda', 'Laksevåg'): 14, ('Ytrebygda', 'Ytrebygda'): 0.1, ('Ytrebygda', 'Årstad'): 17, ('Årstad', 'Knappskog'): 27, ('Årstad', 'Kolltveit'): 25, ('Årstad', 'Kårtveit'): 35, ('Årstad', 'Telavåg'): 47, ('Årstad', 'Straume'): 22, ('Årstad', 'Knarrevik'): 18, ('Årstad', 'Bildøyna'): 22, ('Årstad', 'Spjeld'): 29, ('Årstad', 'Foldnes'): 26, ('Årstad', 'Brattholmen'): 23, ('Årstad', 'Blomøy'): 50, ('Årstad', 'Hellesøy'): 68, ('Årstad', 'Arefjord'): 22, ('Årstad', 'Ebbesvik'): 26, ('Årstad', 'Landro'): 36, ('Årstad', 'Hjelteryggen'): 21, ('Årstad', 'Skogsvåg'): 39, ('Årstad', 'Kleppestø'): 22, ('Årstad', 'Solsvik'): 38, ('Årstad', 'Rongøy'): 44, ('Årstad', 'Hammarsland'): 37, ('Årstad', 'Træsneset'): 43, ('Årstad', 'Bergenhus'): 13, ('Årstad', 'Laksevåg'): 15, ('Årstad', 'Ytrebygda'): 16, ('Årstad', 'Årstad'): 0.1}



df1 = pd.read_excel(filename)


Q_k = {}
A_k1 = {}
A_k2 = {}


def add_parameters():
    """ Use driver and passenger information from json. files to add Parameters
        :return:
        """
    for drivers in drivers_json:
        o_k[drivers_json[drivers]['id']] = drivers_json[drivers]['id']
        d_k[drivers_json[drivers]['id']] = drivers_json[drivers]['id'] + nr_passengers * 2 + nr_drivers
        T_k[drivers_json[drivers]['id']] = drivers_json[drivers]['max_ride_time'] * 1.9/1.5
        Q_k[drivers_json[drivers]['id']] = drivers_json[drivers]['max_capacity']
        A_k1[drivers_json[drivers]['id'] + nr_passengers * 2 + nr_drivers] = drivers_json[drivers]['lower_tw']
        A_k2[drivers_json[drivers]['id'] + nr_passengers * 2 + nr_drivers] = drivers_json[drivers]['upper_tw']
    for passengers in passengers_json:
        T_k[passengers_json[passengers]['id']] = passengers_json[passengers]['max_ride_time'] * 1.9/1.5
        A_k1[passengers_json[passengers]['id'] + nr_passengers] = passengers_json[passengers]['lower_tw']
        A_k2[passengers_json[passengers]['id'] + nr_passengers] = passengers_json[passengers]['upper_tw']

add_parameters()

driver_origin_nodes = {k: o_k[k] for k in D}
driver_destination_nodes = {k: d_k[k] for k in D}



def create_Tij():
    t_ij = {}
    for i, j in A:
        if i in D and j in NP:
            stedsnavn1 = drivers_json["D"+str(i)]["origin_location"]
            stedsnavn2 = passengers_json["P" + str(j)]["origin_location"]
            distance = distance_matrix[(stedsnavn1, stedsnavn2)]
            t_ij[(i,j)] = distance
        if i in D and j == (i + nr_drivers + (2*nr_passengers)):
            stedsnavn1 = drivers_json["D"+str(i)]["origin_location"]
            stedsnavn2 = drivers_json["D"+str(i)]["destination_location"]
            distance = distance_matrix[(stedsnavn1, stedsnavn2)]
            t_ij[(i,j)] = distance
        if i in NP and j in NP:
            stedsnavn1 = passengers_json["P" + str(i)]["origin_location"]
            stedsnavn2 = passengers_json["P" + str(j)]["origin_location"]
            distance = distance_matrix[(stedsnavn1, stedsnavn2)]
            t_ij[(i,j)] = distance
        if i in NP and j in ND:
            stedsnavn1 = passengers_json["P" + str(i)]["origin_location"]
            stedsnavn2 = passengers_json["P" + str(j - nr_passengers)]["destination_location"]
            distance = distance_matrix[(stedsnavn1, stedsnavn2)]
            t_ij[(i,j)] = distance
        if i in ND and j in ND:

            stedsnavn1 = passengers_json["P" + str(i - nr_passengers)]["destination_location"]
            stedsnavn2 = passengers_json["P" + str(j - nr_passengers)]["destination_location"]
           
            distance = distance_matrix[(stedsnavn1, stedsnavn2)]
            t_ij[(i,j)] = distance
        if i in ND and j in list(driver_destination_nodes.values()):
            stedsnavn1 = passengers_json["P" + str(i - nr_passengers)]["destination_location"]
            stedsnavn2 = drivers_json["D" + str(j - nr_drivers - (2*nr_passengers))]["destination_location"]
            distance = distance_matrix[(stedsnavn1, stedsnavn2)]
            t_ij[(i,j)] = distance
           

    return t_ij

print(create_Tij())


def initialize_big_M():
    result={}
    for driver in D:
        result[driver] = T_k[driver] * 2.5
    return result


M = initialize_big_M()

