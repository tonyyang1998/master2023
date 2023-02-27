import numpy as np
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import matplotlib.pyplot as plt
import json
import time
import TestExcel as te
import xlwt
from xlwt import Workbook
import distance_matrix 


filename = "test_instance.xlsx"
#file_to_save = 'Results/Small Instances 1,5/Small_4,1.5.xls'

te.main(filename)
start_time = time.time()
passengers_json = json.load(open('sample_passenger.json'))
drivers_json = json.load(open('sample_driver.json'))

rnd = np.random
rnd.seed(0)

nr_passengers = len(passengers_json)
nr_drivers = len(drivers_json)

'''Sets'''
D = [i for i in range(nr_drivers)]
all_node_index = [i for i in range(nr_passengers * 2 + nr_drivers * 2)]
PP = all_node_index[int(len(D)):int(len(all_node_index) / 2)]
PD = all_node_index[int(len(all_node_index) / 2):-int(len(D))]
P = PP + PD

location_index = {"Knappskog": 1, "Kolltveit":2, "Blomøy":3, "Hellesøy":4, "Foldnes":5,"Brattholmen":6,"Arefjord":7, "Ebbesvik":8,"Straume":9,"Spjeld":10,"Landro":11, "Knarrevik":12,"Hjelteryggen":13,"Skogsvåg":14,"Kleppestø":15,"Solsvik":16,"Rongøy":17,"Hammarsland":18,"Telavåg":19,"Træsneset":20,"Tofterøy":21,"Bildøyna":22,"Kårtveit":23, "Bergenshus":24, "Laksevåg":25, "Ytrebydga":26, "Årstad": 27}
index_location = {y: x for x, y in location_index.items()}


def passenger_candidate_pickup_location_initialization():
    all_location_pairs = distance_matrix.distance_matrix.keys()
    result = {}
    
    for passenger in passengers_json:
        candidate_locations = []
        for location_pair in all_location_pairs:
            if location_pair[0] == passengers_json[passenger]["origin_location"] and location_pair[0]!=location_pair[1]:
                if distance_matrix.distance_matrix[location_pair] <= 10:
                    candidate_locations.append(location_pair)
        result[passengers_json[passenger]["id"]] = candidate_locations
    return result

def initialize_MPi():
    MP_i = {}
    locations =  passenger_candidate_pickup_location_initialization()
    for passenger in locations:
        result = []
        for location_pair in locations[passenger]:
            result.append(location_index[location_pair[1]])
        MP_i[passenger] = result
    return MP_i

MP_i = initialize_MPi()

#print(MP_i)
MD_i = {i: 0 for i in PP}

def initialize_NP():
    NP = []
    for passengers in PP:
        for candidate_locations in MP_i[passengers]:
            NP.append((passengers, candidate_locations))
        if (passengers, 0) not in NP:
            NP.append((passengers, 0))
    return NP

def initialize_ND():
    """Må bearbeides"""
    ND = []
    for passengers in PD:
            ND.append((passengers, MD_i[passengers-nr_passengers]))
    return ND

NP = initialize_NP()
ND = initialize_ND()
NR = NP + ND


'''Parameters'''
o_k = {k:(k, 0) for k in D}
d_k = {k:(k + 2*nr_passengers + nr_drivers, 0) for k in D}
T_k = {}

driver_origin_nodes = {k: o_k[k] for k in D}
driver_destination_nodes = {k: d_k[k] for k in D}

print(MP_i)

def initialize_Ak():
    result = {}
    Ak = {k: [((i,m),(j,n)) for (i,m) in NR + [o_k[k]] for (j,n) in NR + [d_k[k]] if ((i,m)!=(j,n))] for k in D}
    for driver in Ak:
        all_ar = list(Ak[driver])
        all_arcs = all_ar
        for arc in all_arcs:
            """Remove all arcs where (i,m) is a pick up node and (j,n) is driver destination"""
            if arc[0] in NP and arc[1] in list(d_k.values()):
                all_arcs.remove(arc)

            """Remove all arcs where (i,m) is a driver origin and (j,n) is a delivery node"""
            if (arc[0] in list(o_k.values()) and arc[1] in ND) or (arc[0][0] in list(o_k.keys()) and arc[1][0] in PD) or (arc == ((0, 0), (6, 0))):
                all_arcs.remove(arc)
            
            """Remove all arcs where between candidate locations"""
            if (arc[0][0]==arc[1][0]):
                print(arc[0][0],arc[1][0] )
                all_arcs.remove(arc)
                
            
            """Remove all arcs where (i,m) is a delivery and (j,n) is a pick up node"""
            if arc[0] in ND and arc[1] in NP:
                all_arcs.remove(arc)
        print(all_arcs)

        
        result[driver] = all_arcs

   
        

print(initialize_Ak())

def initialize_Timjn():
    """IKKE ferdig"""
    T_imjn = {}
    for node in NR + list(o_k.values()):
        print(node)
        if node[0] in D and node[1] in PP:
            stedsnavn1 = drivers_json["D"+str(node[0])]["origin_location"]
            stedsnavn2 = passengers_json["P" + str(node[1])]["origin_location"]
            print(stedsnavn1)
            print(stedsnavn2)

    
#print(initialize_Timjn())

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
        T_k[drivers_json[drivers]['id']] = drivers_json[drivers]['max_ride_time'] * 1.5
        Q_k[drivers_json[drivers]['id']] = drivers_json[drivers]['max_capacity']
        A_k1[drivers_json[drivers]['id'] + nr_passengers * 2 + nr_drivers] = drivers_json[drivers]['lower_tw']
        A_k2[drivers_json[drivers]['id'] + nr_passengers * 2 + nr_drivers] = drivers_json[drivers]['upper_tw']
    for passengers in passengers_json:
        T_k[passengers_json[passengers]['id']] = passengers_json[passengers]['max_ride_time'] * 1.9/1.5
        A_k1[passengers_json[passengers]['id'] + nr_passengers] = passengers_json[passengers]['lower_tw']
        A_k2[passengers_json[passengers]['id'] + nr_passengers] = passengers_json[passengers]['upper_tw']

add_parameters()

"""Helper functions"""

delivery_and_pickup_node_pairs = {PD[i]: PP[i] for i in range(len(PD))}
pickup_and_delivery_node_pairs = {PP[i]: PD[i] for i in range(len(PD))}




driver_origin_nodes = {k: o_k[k] for k in D}
driver_destination_nodes = {k: d_k[k] for k in D}


def initialize_big_M():
    result={}
    for driver in D:
        result[driver] = T_k[driver] * 2.5
    return result


M = initialize_big_M()

