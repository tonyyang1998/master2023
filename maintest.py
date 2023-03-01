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



def initialize_M_i():
    Mi = {}
    for passenger in PP:
        Mi[passenger] = MP_i[passenger] + [MD_i[passenger]]
    return Mi
M_i = initialize_M_i()


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

driver_origin_nodes = {k: o_k[k] for k in D}
driver_destination_nodes = {k: d_k[k] for k in D}


N = list(driver_origin_nodes.values()) + NR + list(driver_destination_nodes.values())


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
            
        new_edges=[]
        for edge in all_arcs:
            """Remove all arcs where between candidate locations"""
            if edge[0][0] != edge[1][0]:
                new_edges.append(edge)
                
        """Remove all arcs where (i,m) is a delivery and (j,n) is a pick up node"""
        new_edges1 = [edge for edge in new_edges if not (edge[0][0] >= nr_drivers + nr_passengers and edge[1][0] <= nr_drivers + nr_passengers - 1)]
        """Remove all arcs where (i,m) is a driver origin and (j,n) is a delivery node"""
        new_edges2 = [edge for edge in new_edges1 if not (edge[0][0] in D and edge[1][0] in PD)]
        result[driver] = new_edges2
    return result

A_k = initialize_Ak()


def initialize_Timjn():
    """IKKE ferdig"""
    T_imjn = {}
   
    for driver in A_k:
        for arc in A_k[driver]:
            """Between driver origin and and destination"""
           
            if arc[0][0] in D and arc[1][0] in d_k[driver]:
                stedsnavn1 = drivers_json["D" + str(arc[0][0])]["origin_location"]
                stedsnavn2 = drivers_json["D" + str(arc[0][0])]["destination_location"]
                distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                T_imjn[arc] = distance

            """Between driver origin and all pick up candidate locations """
            if arc[0][0] in D and arc[1][0] in PP:
                """If pick up node is (j, x)"""
                
                if arc[1][1]!=0:
                    stedsnavn1 = drivers_json["D" + str(arc[0][0])]["origin_location"]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                else:   
                    """If pick up node is (j, 0)""" 
                    stedsnavn1 = drivers_json["D" + str(arc[0][0])]["origin_location"]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0])]["origin_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance

            """Between all pick up nodes"""
            if arc[0][0] in PP and arc[1][0] in PP:
                """Between origins"""
                if arc[0][1] == 0 and arc[1][1] == 0:      
                    stedsnavn1 = passengers_json["P" + str(arc[0][0])]["origin_location"]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0])]["origin_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    
                    T_imjn[arc] = distance
                """From origin to candidate pick up """
                if arc[0][1] == 0 and arc[1][1] != 0:
                    stedsnavn1 = passengers_json["P" + str(arc[0][0])]["origin_location"]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    
                    T_imjn[arc] = distance
                """Between candidate pick ups"""
                if arc[0][1] != 0 and arc[1][1] != 0:
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                   
                    T_imjn[arc] = distance
                """From candidate pick up to origin"""
                if arc[0][1] != 0 and arc[1][1] == 0:
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0])]["origin_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance

            """Between pick up and delivery nodes"""
            if arc[0][0] in PP and arc[1][0] in PD:
                """Between origin and destination"""
                if arc[0][1] == 0 and arc[1][1] == 0:      
                    stedsnavn1 = passengers_json["P" + str(arc[0][0])]["origin_location"]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0] - nr_passengers)]["destination_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                """From origin to candidate delivery (Ingen candidate deliveries atm) """
                if arc[0][1] == 0 and arc[1][1] != 0:
                    stedsnavn1 = passengers_json["P" + str(arc[0][0])]["origin_location"]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
            
                    T_imjn[arc] = distance
                """Between candidate pick up to candidate delivery"""
                
                if arc[0][1] != 0 and arc[1][1] != 0:
         
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                """From candidate pick up to destination"""
                if arc[0][1] != 0 and arc[1][1] == 0:
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0] - nr_passengers)]["destination_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance

            """Between all delivery nodes"""
            if arc[0][0] in PD and arc[1][0] in PD:
                """Between destinations"""
                if arc[0][1] == 0 and arc[1][1] == 0:      
                    stedsnavn1 = passengers_json["P" + str(arc[0][0] - nr_passengers)]["destination_location"]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0] - nr_passengers)]["destination_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                """From destination to candidate delivery (Ingen candidate deliveries atm) """
                if arc[0][1] == 0 and arc[1][1] != 0:
                    stedsnavn1 = passengers_json["P" + str(arc[0][0] - nr_passengers)]["destination_location"]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                """Between candidate pick up and candidate deliveries"""
                if arc[0][1] != 0 and arc[1][1] != 0:
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = index_location[arc[1][1]]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                """From candidate delivery to destination"""
                if arc[0][1] != 0 and arc[1][1] == 0:
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = passengers_json["P" + str(arc[1][0] - nr_passengers)]["destination_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance

            """From candidate delivery to driver destination"""
            if arc[0][0] in PD and arc[1][0] in d_k[driver]:
                """Between passenger destinations and driver destination"""
                if arc[0][1] == 0 and arc[1][1] == 0:      
                    stedsnavn1 = passengers_json["P" + str(arc[0][0] - nr_passengers)]["destination_location"]
                    stedsnavn2 = drivers_json["D" + str(arc[1][0] - nr_drivers - 2*nr_passengers)]["destination_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
                """From candidate delivery to driver destination"""
                if arc[0][1] != 0 and arc[1][1] == 0:
                    stedsnavn1 = index_location[arc[0][1]]
                    stedsnavn2 = drivers_json["D" + str(arc[1][0] - nr_drivers - 2*nr_passengers)]["destination_location"]
                    distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                    T_imjn[arc] = distance
    return T_imjn

T_imjn = initialize_Timjn()


def initialize_Tim():
    Tim = {}
    visited_nodes = []
    arcs = list(T_imjn.keys())
    for arc in arcs:
        node1 = arc[0]
        node2 = arc[1]
        if node1 not in visited_nodes:
            """If pick up node1 is (i, 0)"""
            if node1[0] in PP and node1[1] == 0:
                Tim[node1] = 0.1 
                visited_nodes.append(node1)
            """If pick up node1 is (i, x)"""
            if node1[0] in PP and node1[1] != 0:
                stedsnavn1 = passengers_json["P" + str(node1[0])]["origin_location"]
                stedsnavn2 = index_location[node1[1]]
                distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                Tim[node1] = distance
                visited_nodes.append(node1)

            """If delivery node1 is (j, 0)"""
            if node1[0] in PD and node1[1] == 0:
                Tim[node1] = 0.1 
                visited_nodes.append(node1)
            """If delivery node1 is (j, x)"""
            if node1[0] in PD and node1[1] != 0:
                stedsnavn1 = passengers_json["P" + str(node1[0] - nr_passengers)]["destination_location"]
                stedsnavn2 = index_location[node1[1]]
                distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                Tim[node1] = distance
                visited_nodes.append(node1)

        if node2 not in visited_nodes:
            """If pick up node2 is (i, 0)"""
            if node2[0] in PP and node2[1] == 0:
                Tim[node2] = 0.1 
                visited_nodes.append(node2)
            """If pick up node2 is (i, x)"""
            if node2[0] in PP and node2[1] != 0:
                stedsnavn1 = passengers_json["P" + str(node2[0])]["origin_location"]
                stedsnavn2 = index_location[node2[1]]
                distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                Tim[node2] = distance
                visited_nodes.append(node2)

            """If delivery node2 is (j, 0)"""
            if node2[0] in PD and node2[1] == 0:
                Tim[node2] = 0.1 
                visited_nodes.append(node2)
            """If delivery node2 is (j, x)"""
            if node2[0] in PD and node2[1] != 0:
                stedsnavn1 = passengers_json["P" + str(node2[0] - nr_passengers)]["destination_location"]
                stedsnavn2 = index_location[node2[1]]
                distance = distance_matrix.distance_matrix[(stedsnavn1, stedsnavn2)]
                Tim[node2] = distance
                visited_nodes.append(node2)
    return Tim

T_im = initialize_Tim()


A_i1 = {}
A_i2 = {}
Q_k = {}
T_k = {}

def add_parameters():
    """ Use driver and passenger information from json. files to add Parameters
        :return:
        """
    for drivers in drivers_json:
        T_k[drivers_json[drivers]['id']] = drivers_json[drivers]['max_ride_time'] * 1.5
        Q_k[drivers_json[drivers]['id']] = drivers_json[drivers]['max_capacity']
        A_i1[drivers_json[drivers]['id']] = drivers_json[drivers]['lower_tw']
        A_i2[drivers_json[drivers]['id']] = drivers_json[drivers]['upper_tw']
    for passengers in passengers_json:
        T_k[passengers_json[passengers]['id']] = passengers_json[passengers]['max_ride_time'] * 1.9/1.5
        A_i1[passengers_json[passengers]['id']] = passengers_json[passengers]['lower_tw']
        A_i2[passengers_json[passengers]['id']] = passengers_json[passengers]['upper_tw']

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



"""Variables"""

model = Model('RRP')

def set_variables():
    x_kimjn = model.addVars([(k, i, m, j, n) for k in D for (i, m) in NR for (j, n) in NR], vtype=GRB.BINARY, name='x_kimjn')
    model.update()
    xs_kim = model.addVars([(k, i, m) for k in D for (i, m) in NP], vtype=GRB.BINARY, name='xs_kim')
    model.update()
    xe_kjn = model.addVars([(k, j, n) for k in D for (j, n) in ND], vtype=GRB.BINARY, name='xe_kjn')
    model.update()
    xod_k = model.addVars([k for k in D], vtype=GRB.BINARY, name='xod_k')
    model.update()
    y_im = model.addVars([(i, m) for (i, m) in NR], vtype=GRB.BINARY, name='y_im')
    model.update()
    z_ki = model.addVars([(k, i) for k in D for i in PP], vtype=GRB.BINARY, name='z_ki')
    model.update()
    t_kim = model.addVars([(k, i, m) for k in D for (i, m) in N], vtype=GRB.CONTINUOUS, name='t_ki')
    model.update()
    return x_kimjn, xs_kim, xe_kjn, xod_k, y_im, z_ki, t_kim

x_kimjn, xs_kim, xe_kjn, xod_k, y_im, z_ki, t_kim = set_variables()

#print(t_kim)

"""Objective"""
def set_objective():
    model.ModelSense = GRB.MAXIMIZE
    model.setObjectiveN(quicksum(z_ki[k, i] for k in D for i in PP), index = 0, priority = 1)
    model.setObjectiveN(- quicksum((t_kim[k, d_k[k][0], d_k[k][1]] - (t_kim[k, o_k[k][0], o_k[k][1]]) for k in D)) - quicksum((t_kim[k, nr_passengers + i, 0] - t_kim[k, i, 0]) for k in D for i in PP), index = 1, priority = 0)
    model.update()

set_objective()


"""Constraints"""


def add_constraints():
    '''Routing constraits'''
    model.addConstrs(quicksum(xs_kim[k, i, m] for (i, m) in NP) + xod_k[k] == 1 for k in D)
    model.addConstrs(quicksum(xe_kjn[k, j, n] for (j, n) in ND) + xod_k[k] == 1 for k in D)

    model.addConstrs(xs_kim[k, i, m] + quicksum(x_kimjn[k, j, n, i, m] for (j, n) in NP) == quicksum(x_kimjn[k, i, m, j, n] for (j, n) in NR) for k in D for (i, m) in NP)
    model.addConstrs(xe_kjn[k, j, n] == quicksum(x_kimjn[k, i, m, j, n] for (i,m) in NR) for k in D for (j, n) in ND)

    model.addConstrs(quicksum(xs_kim[k, i, m] for k in D) + quicksum(x_kimjn[k, j, n, i, m] for k in D for (j, n) in NR) - y_im[i, m] == 0 for (i, m) in NP)
    model.addConstrs(quicksum(y_im[i, m] for m in M_i[i]) <= 1 for i in PP)

    model.addConstrs(quicksum(xs_kim[k, i, m] for m in MP_i[i]) + quicksum(x_kimjn[k, i, m, j, n] for m in MP_i[i] for (j, n) in NR) == z_ki[k, i] for k in D for i in PP)

    model.addConstrs(xod_k[k] <= 1 - z_ki[k, i] for k in D for i in PP)

    """Coupling and precedence constraints"""
    model.addConstrs(quicksum(x_kimjn[k, i, m, j, n] for (j, n) in NR for m in MP_i[i]) - quicksum(x_kimjn[k, j, n, nr_passengers + i, m] for (j,n) in NR for m in [MD_i[i]]) == 0 for k in D for i in PP)
    model.addConstrs(t_kim[k, i, m] + T_imjn[(i, m), (nr_passengers + i, n)] - t_kim[k, nr_passengers + i, n] <= 0 for k in D for (i, m) in NP for o in PP for n in [MD_i[o]])

    """Time constraint"""
    model.addConstrs(
        t_kim[k, i, m] + T_imjn[(i, m), (j, n)] - t_kim[k, j, n] - M[k] *(1 - x_kimjn[k, i, m, j, n]) <= 0 for k in D for (i, m) in NR for (j, n) in NR if ((i, m), (j, n)) in A_k[k])
    model.addConstrs(
        t_kim[k, i, m] + T_imjn[(i, m), (j, n)] - t_kim[k, j, n] + M[k] *(1 - x_kimjn[k, i, m, j, n]) >= 0 for k in D for (i, m) in NR for (j, n) in NR if ((i, m), (j, n)) in A_k[k])

    model.addConstrs(
        t_kim[k, o_k[k][0], o_k[k][1]] + T_imjn[(o_k[k][0],o_k[k][1]), (i, m)] - t_kim[k, i, m] - M[k] *(1 - xs_kim[k, i, m]) <= 0 for k in D for (i, m) in NP)
    model.addConstrs(
        t_kim[k, o_k[k][0], o_k[k][1]] + T_imjn[(o_k[k][0],o_k[k][1]), (i, m)] - t_kim[k, i, m] + M[k] *(1 - xs_kim[k, i, m]) >= 0 for k in D for (i, m) in NP)

    model.addConstrs(
        t_kim[k, i, m] + T_imjn[(i, m), (d_k[k][0], d_k[k][1])] - t_kim[k, d_k[k][0], d_k[k][1]] - M[k] *(1 - xe_kjn[k, i, m]) <= 0 for k in D for (i, m) in ND)
    model.addConstrs(
        t_kim[k, i, m] + T_imjn[(i, m), (d_k[k][0], d_k[k][1])] - t_kim[k, d_k[k][0], d_k[k][1]] + M[k] *(1 - xe_kjn[k, i, m]) >= 0 for k in D for (i, m) in ND)

    model.addConstrs(A_i1[i] <= t_kim[k, nr_passengers + i, n] + T_im[nr_passengers, n] for k in D for i in PP for n in [MD_i[i]])
    model.addConstrs(t_kim[k, nr_passengers + i, n] + T_im[nr_passengers, n] <= A_i2[nr_passengers] for k in D for i in PP for n in [MD_i[i]])
    
    model.addConstrs(A_i1[k] <= t_kim[k, d_k[k][0], d_k[k][1]] for k in D)
    model.addConstrs(t_kim[k, d_k[k][0], d_k[k][1]] <= A_i2[k] for k in D)

    disposable1 = model.addConstrs(t_kim[k, nr_passengers + i, 0] - t_kim[k, i, 0] <= T_k[i] for k in D for i in PP)
    disposable2 = model.addConstrs(t_kim[k, d_k[k][0], d_k[k][1]] - t_kim[k, o_k[k][0], o_k[k][1]] <= T_k[k] for k in D)

    model.addConstrs(t_kim[k, i, 0] <= t_kim[k, i, m] - (T_im[i, m] * y_im[i, m]) for k in D for i in PP for m in MP_i[i])

    print(T_im)
    print(y_im)
    model.addConstrs(t_kim[k, nr_passengers + i, 0] >= t_kim[k, nr_passengers + i, m] + (T_im[nr_passengers + i, m] * y_im[nr_passengers + i, m]) for k in D for i in PP for m in [MD_i[i]])

    '''Capacity constraint'''
    model.addConstrs(quicksum(z_ki[k, i] for i in PP) <= Q_k[k] for k in D)

    model.update()
    return disposable1, disposable2


"""Optimize"""
def optimize():
    model.setParam('TimeLimit', 3600)
    add_constraints()
    model.optimize(my_callback)


result_solution = []
result_bound = []
result_time = []

def my_callback(model, where):
    
    if where == GRB.Callback.MIP:
        current_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        current_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        if current_best not in result_solution:
            if current_best<0:
                result_solution.append(0)
            else:
                result_solution.append(current_best)
                result_bound.append(current_bound)
                result_time.append(runtime)
        if current_bound not in result_bound:
            result_solution.append(current_best)
            result_bound.append(current_bound)
            result_time.append(runtime)

def debug():
    model.computeIIS()
    model.write('model.MPS')
    model.write('model.lp')
    model.write('model.ilp')


def run_only_once():
    optimize()

run_only_once()