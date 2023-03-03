import numpy as np
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import matplotlib.pyplot as plt
import json
import time
import TestExcel as te
import xlwt
from xlwt import Workbook


filename = "Small Instances 1,5/Small4,1.5.xlsx"
file_to_save = 'Results/Small Instances 1,5/Small_4,1.5.xls'

te.main(filename)

start_time = time.time()

passengers_json = json.load(open('sample_passenger.json'))
drivers_json = json.load(open('sample_driver.json'))


rnd = np.random
rnd.seed(0)

#nr_passengers = len(passengers_json)
#nr_drivers = len(drivers_json)

nr_passengers = 6
nr_drivers = 2

'''Sets'''
D = [i for i in range(nr_drivers)]
all_node_index = [i for i in range(nr_passengers * 2 + nr_drivers * 2)]
PP = all_node_index[int(len(D)):int(len(all_node_index) / 2)]
PD = all_node_index[int(len(all_node_index) / 2):-int(len(D))]
P = PP + PD
print(P)

delivery_and_pickup_node_pairs = {ND[i]: NP[i] for i in range(len(ND))}
pickup_and_delivery_node_pairs = {NP[i]: ND[i] for i in range(len(ND))}

'''Parameters'''
o_k = {}
d_k = {}
T_k = {}
T_ij = {(i, j): np.hypot(xc[i] - xc[j], yc[i] - yc[j]) for i, j in A}

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


def initialize_big_M():
    result={}
    for driver in D:
        result[driver] = T_k[driver] * 2.5
    return result


M = initialize_big_M()


def check_time_window_between_arc(i, j):
    """ Checks if the time of traveling between (i, j) = (driver origin or passenger pick up nodes, passenger delivery node) is within the time window of j
        :param i: Integer - origin or pick up nodes
        :param j: Integer - delivery nodes
        :return: Boolean
        """
    return T_ij[(i, j)] < A_k2[j]


def check_max_ride_time_between_arc(i, j):
    """ Checks if the time of traveling between (i, j) = (driver origin node, passenger delivery or driver destination node) is within the max travel time to j
        :param i: Integer - origin or pick up nodes
        :param j: Integer - delivery nodes
        :return: Boolean
        """
    return T_k[i] < T_ij[(i, j)]


def process_NK():
    '''Removes nodes where:
         1) the quickest path between (i, j) = (driver origin node, passenger delivery node) is not within j's timewindow
         2) the quickest path between (i, j) = (passenger pick up node, passenger delivery node) is not within j's timewindow
         must also include a max_ride_time for passenger here removal
         3) the quickest path between (i, j) = (driver origin node, passenger delivery node) is not within the maximum ridetime from driver origin node i to j
        :return: {k: [nodes]} - returns set of feasible nodes driver k can travel to and excludes other driver's origin and destination nodes
        '''
    NK = {}
    for driver in D:
        resulting_nodes = []
        for arc in T_ij:
            i = arc[0]
            j = arc[1]
            if i not in resulting_nodes:
                resulting_nodes.append(i)
            if j not in resulting_nodes:
                resulting_nodes.append(j)
            if i == driver and j in ND and not check_time_window_between_arc(i, j):
                pickup_node = delivery_and_pickup_node_pairs[j]
                resulting_nodes.remove(pickup_node)
                resulting_nodes.remove(j)
            if i in NP and j in ND and pickup_and_delivery_node_pairs[i] == j and not check_time_window_between_arc(i,
                                                                                                                    j):
                resulting_nodes.remove(i)
                resulting_nodes.remove(j)
            if i == driver and j in ND and not check_max_ride_time_between_arc(i, j):
                pickup_node = delivery_and_pickup_node_pairs[j]
                resulting_nodes.remove(pickup_node)
                resulting_nodes.remove(j)
        resulting_nodes.sort()
        NK[driver] = resulting_nodes
        for node in NK[driver]:
            if node != driver and node in list(driver_origin_nodes.values()):
                resulting_nodes.remove(node)
            if node != driver_destination_nodes[driver] and node in list(driver_destination_nodes.values()):
                resulting_nodes.remove(node)
        NK[driver] = resulting_nodes
    return NK


NK = process_NK()


def generate_NPK(NK):
    """
        :param NK: {k: [nodes]} - set of preprocessed feasible nodes for driver k to visit
        :return: {k: [nodes]} - set of feasible pick up nodes for driver k to visit
        """
    NPK = {}
    for drivers in NK:
        nodes = []
        for node in NK[drivers]:
            if node in NP:
                nodes.append(node)
        NPK[drivers] = nodes
    return NPK


def generate_NDK(NK):
    """
        :param NK: {k: [nodes]} - set of preprocessed feasible nodes for driver k to visit
        :return: {k: [nodes]} - set of feasible pick up nodes for driver k to visit
        """
    NDK = {}
    for drivers in NK:
        nodes = []
        for node in NK[drivers]:
            if node in ND:
                nodes.append(node)
        NDK[drivers] = nodes
    return NDK


NPK = generate_NPK(NK)
NDK = generate_NDK(NK)


def check_driver_origin_node(node):
    """ Checks if node is a driver origin node
        :param node: Integer - a node
        :return: Boolean - True if node is a driver origin node, False otherwise
        """
    return node in driver_origin_nodes.values()


def check_driver_destination_node(node):
    """ Checks if node is a driver destination node
        :param node: Integer - a node
        :return: Boolean - True if node is a driver destination node, False otherwise
        """
    return node in driver_destination_nodes.values()


def from_delivery_to_pickup_arc(arc):
    """  Checks if the arc (i, j) is from a passenger delivery node to a passenger pick up node
        :param arc: (i, j) - arc from node i to node j
        :return: Boolean - returns True if i is a delivery node AND j is a pick up node in arc (i, j), False otherwise
        """
    return arc[0] in ND and arc[1] in NP


def process_AK(NK):
    """ Removes:
        1) all arcs (i, j) where j are origin nodes, i are destination nodes, and i j in (i, j) where i is a pick up node and j is a delivery node
        2) all arcs going in to j, where j is a delivery node and if the ride time for a passenger to travel from its pick up node to j is higher than the maximum ride time for passenger.
                If a passenger cannot get to its delivery node within that persons max ride time from their pick up node using the shortest arc,
                then there should be no arcs going into that persons delivery node because we would never be able to deliver that guy
        :param NK: {k, [nodes]} - set of preprocessed feasible nodes for driver k to visit
        :return: {k, [arcs]} - returns a set of feasible arcs (i, j) for driver k to travel with.
        """
    result = {}
    AK = {k: [(i, j) for i in NK[k] for j in NK[k] if i != j] for k in NK}
    for driver in AK:
        arcs = []
        """Removes:
         1) all arcs where i is a destination node and j is a origin node, and removes all arcs from deliverys to pick ups"""

        for arc in AK[driver]:
            i = arc[0]
            j = arc[1]
            if not check_driver_origin_node(j):
                if not check_driver_destination_node(i):
                    if not from_delivery_to_pickup_arc(arc):
                        arcs.append(arc)

        for arc in AK[driver]:
            """Removes
                2) all arcs going into j, if a passenger that has j as the delivery node 
                where traveling from that guys pick up node to j exceeds the max time a passenger is willing to ride
                3) all arcs where i is a pick up node and j is a destination node
                4) all arcs that goes from origin node to a delivery node
                5) all arcs from origin to destination where j is not the assocaited destination node
                6) all arcs where i is not the origin node of driver k
                7) all arcs where i is a delivery node and j is not the associated destination node for driver k
                8) (i,j) and (j, n+i) where i is a pick up node and j is a pick up node, where the time required to travel (i, j) plus
                (j, n+i) is longer than the max ridetime of i"""
            i = arc[0]
            j = arc[1]
            if i in NP and j in ND and pickup_and_delivery_node_pairs[i] == j and T_k[i] < T_ij[(i, j)]:
                arcs.remove(arc)
                for i in D + NP:
                    if (i, j) in arcs:
                        arcs.remove((i, j))
                # legge til å fjerne alle kanter som går inn til den pick up noden også

            if i != driver_origin_nodes[driver] and i in NP:
                if j in list(driver_destination_nodes.values()):
                    arcs.remove((i, j))

            if i == driver_origin_nodes[driver] and j in ND:
                arcs.remove((i, j))


            if i == driver_origin_nodes[driver] and j in list(driver_destination_nodes.values()):
                if j != driver_destination_nodes[driver]:
                    arcs.remove((i, j))

            if i != driver_origin_nodes[driver] and i in list(driver_origin_nodes.values()):
                if j in NP or j in ND or j in list(driver_destination_nodes.values()):
                    arcs.remove((i, j))

            if i in ND:
                if j != driver_destination_nodes[driver]:
                    if j in list(driver_destination_nodes.values()):
                        arcs.remove((i, j))

            if i in NP:
                if j in NP:
                    if i!=j:
                        if T_ij[(i,j)] + T_ij[(j, nr_passengers + i)] > T_k[i]:
                            arcs.remove((i, j))
                            arcs.remove((j, nr_passengers+i))
        result[driver] = arcs

    return result

AK = process_AK(NK)

model = Model('RRP')



def set_variables():
    """Add variables"""
    x = model.addVars([(k, i, j) for k in D for (i, j) in AK[k]], vtype=GRB.BINARY, name='x_kij')
    model.update()
    z = model.addVars(NP, vtype=GRB.BINARY, name='z_i')
    model.update()
    t = model.addVars([(k, i) for k in D for i in NK[k]], vtype=GRB.CONTINUOUS, name='t_ki')
    model.update()
    return x, z, t


x, z, t = set_variables()


def set_objective1():
    '''Model'''
    #model.setObjective(quicksum(T_ij[i, j] * x[k, i, j] for k in D for i in NK[k] for j in NK[k] if (i, j) in AK[k]),GRB.MINIMIZE)
    # model.setObjective(quicksum(T_ij[i,j]*x[k,i,j] for i in N for j in N for k in D if i!=j), GRB.MINIMIZE)
    model.modelSense = GRB.MAXIMIZE
    model.setObjective(quicksum(z[i] for i in NP))
    model.update()


set_objective1()


nodes_without_destinations = {}
for k in D:
    liste = []
    for i in NK[k]:
        destinations = list(driver_destination_nodes.values())
        if i not in destinations:
            liste.append(i)
    nodes_without_destinations[k] = liste


def add_constraints():
    '''Constraints'''
    '''Routing constraits'''

    model.addConstrs(
        quicksum(x[k, i, j] for j in NPK[k] + [driver_destination_nodes[k]] if (i, j) in AK[k]) == 1 for k in D for i in
        [driver_origin_nodes[k]])
    model.addConstrs(
        quicksum(x[k, i, j] for i in [driver_origin_nodes[k]] + NDK[k] if (i, j) in AK[k]) == 1 for k in D for j in
        [driver_destination_nodes[k]])

    model.addConstrs(((quicksum(x[k, i, j] for j in NK[k] if j not in [driver_origin_nodes[k]] if (i, j) in AK[k]))
                      == quicksum(x[k, j, i] for j in NK[k] if j not in [driver_destination_nodes[k]] if (j, i) in AK[k])) for k in D for
                     i in NK[k] if i not in [driver_origin_nodes[k]] if i not in [driver_destination_nodes[k]])

    model.addConstrs((quicksum(x[k, i, j] for j in NK[k] if (i, j) in AK[k])) -
                     (quicksum(x[k, nr_passengers + i, j] for j in ND + [driver_destination_nodes[k]] if
                               (i + nr_passengers, j) in AK[k])) == 0 for k in D for i in NPK[k])

    model.addConstrs(
        (quicksum(x[k, i, j] for k in D for j in NK[k] if j not in list(driver_origin_nodes.values()) if
        (i, j) in AK[k])) - z[i] == 0 for i in NP)

    model.addConstrs(
        (quicksum(x[k, i, j] for k in D for i in NK[k] if i not in list(driver_destination_nodes.values()) if
        (i, j) in AK[k])) <= 1 for j in NP + ND)


    '''Precedence constraint'''
    model.addConstrs(t[k, i] + T_ij[i, nr_passengers + i] - t[k, nr_passengers + i] <= 0 for k in D for i in NP if
                     (i, i + nr_passengers) in AK[k])

    '''Time constraint'''
    model.addConstrs(
        t[(k, i)] + T_ij[(i, j)] - t[(k, j)] - M[k] *(1 - x[k, i, j]) <= 0 for k in D for i in NK[k] for j in NK[k] if
        (i, j) in AK[k])

    model.addConstrs(A_k1[nr_passengers + i] <= t[k, nr_passengers + i] for k in D for i in NPK[k])
    model.addConstrs(t[k, nr_passengers + i] <= A_k2[nr_passengers + i] for k in D for i in NPK[k])

    model.addConstrs(A_k1[driver_destination_nodes[k]] <= t[k, driver_destination_nodes[k]] for k in D)
    model.addConstrs(t[k, driver_destination_nodes[k]] <= A_k2[driver_destination_nodes[k]] for k in D)



    #new waiting constraint

    model.addConstrs(
        x[k, i, j] * (t[k, i] + T_ij[i, j] - t[k, j]) == 0 for k in D for i in NK[k] for j in NK[k] if
        (i, j) in AK[k])

    model.addConstrs(
        t[(k, i)] + T_ij[(i, j)] - t[(k, j)] + M[k] * (1 - x[k, i, j]) >= 0 for k in D for i in NK[k] for j in NK[k] if
        (i, j) in AK[k])

    '''Capacity constraint'''
    model.addConstrs(
        quicksum(x[k, i, j] for i in NPK[k] for j in NK[k] if (i, j) in AK[k]) <= Q_k[k] for k in D)

    model.update()
    disposable = model.addConstrs(t[k, nr_passengers + i] - t[k, i] <= T_k[i] for k in D for i in NPK[k])
    model.update()
    endaen = model.addConstrs(t[k, driver_destination_nodes[k]] - t[k, driver_origin_nodes[k]] <= T_k[k] for k in D)

    model.update()
    return disposable, endaen


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


"""Visualization & debug"""
def debug():
    model.computeIIS()
    model.write('model.MPS')
    model.write('model.lp')
    model.write('model.ilp')


def sort_path(arcs):
    sorted_path = {}
    path={}
    for driver in arcs:
        path1 = []
        b = []
        a = arcs[driver]
        d = {t[0]: t[1] for t in a}
        prev_key = driver
        for i in range(len(d)):
            path1.append(prev_key)
            b.append((prev_key, d[prev_key]))
            prev_key = d[prev_key]
        sorted_path[driver] = b
        path[driver] = path1

    picked_up = {}
    for driver in path:
        picked_up_passengers = []
        for passenger in path[driver]:
            if passenger in NP:
                picked_up_passengers.append(passenger)
        picked_up[driver] = picked_up_passengers

    return sorted_path, path, picked_up


def visualize():
    arcs = {}
    arcsum = {}
    plt.figure(1)
    for k in D:
        active_arcs = [a for a in AK[k] if x[k, a[0], a[1]].x > 0.99]
        arc_sum = 0
        for i, j in active_arcs:
            plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='g', zorder=0)

        for i in range(len(active_arcs)):
            arc_sum += T_ij[active_arcs[i]]
        arcs[k] = active_arcs
        arcsum[k] = arc_sum

    driver_origin_coordinates_x = []
    driver_destination_coordinates_x = []
    passenger_pick_up_coordinates_x = []
    passenger_delivery_coordinates_x = []
    driver_origin_coordinates_y = []
    driver_destination_coordinates_y = []
    passenger_pick_up_coordinates_y = []
    passenger_delivery_coordinates_y = []

    """Plot driver origin nodes"""
    for driver in range(nr_drivers):
        driver_origin_coordinates_x.append(xc[driver])
        driver_origin_coordinates_y.append(yc[driver])

    """Plot passenger pick up nodes"""
    for passenger in range(nr_drivers, nr_passengers + nr_drivers):
        passenger_pick_up_coordinates_x.append(xc[passenger])
        passenger_pick_up_coordinates_y.append(yc[passenger])

    # Plot passenger delivery nodes
    for passenger in range(nr_drivers + nr_passengers, nr_passengers * 2 + nr_drivers):
        passenger_delivery_coordinates_x.append(xc[passenger])
        passenger_delivery_coordinates_y.append(yc[passenger])

    # Plot driver destination nodes
    for driver in range(nr_passengers * 2 + nr_drivers, nr_passengers * 2 + nr_drivers * 2):
        driver_destination_coordinates_x.append(xc[driver])
        driver_destination_coordinates_y.append(yc[driver])

    plt.scatter(driver_origin_coordinates_x, driver_origin_coordinates_y, c='r', marker='s',
                label='Driver origin node')
    plt.scatter(passenger_pick_up_coordinates_x, passenger_pick_up_coordinates_y, c='b', marker='o',
                label='Passenger pick up node')
    plt.scatter(passenger_delivery_coordinates_x, passenger_delivery_coordinates_y, c='c', marker='o',
                label='Passenger delivery node')
    plt.scatter(driver_destination_coordinates_x, driver_destination_coordinates_y, c='m', marker='s',
                label='Driver destination node')
    i = 0
    for z, y in zip(driver_origin_coordinates_x, driver_origin_coordinates_y):
        i = i + 1
        label = 'DO' + str(i)
        plt.annotate(label, (z, y), ha='center')
    i = 0
    for z, y in zip(passenger_pick_up_coordinates_x, passenger_pick_up_coordinates_y):
        i = i + 1
        label = 'PP' + str(i)
        plt.annotate(label, (z, y), ha='center')
    i = 0
    for z, y in zip(passenger_delivery_coordinates_x, passenger_delivery_coordinates_y):
        i = i + 1
        label = 'PD' + str(i)
        plt.annotate(label, (z, y), ha='center')
    i = 0
    for z, y in zip(driver_destination_coordinates_x, driver_destination_coordinates_y):
        i = i + 1
        label = 'DD' + str(i)
        plt.annotate(label, (z, y), ha='center')

    plt.legend()
    #plt.show()


    arcs, path, picked_up = sort_path(arcs)


    return arcs, path, picked_up



def create_pareto_front():
    objective_values = {}
    for epsilon in range(0, 11):
        disposable, endaen = add_constraints()
        model.optimize()
        objective = model.getObjective()
        get_feasible_variables()
        objective_values[epsilon] = objective.getValue()
        visualize()
        model.remove(disposable)
        model.remove(endaen)
        model.update()

    plt.scatter(list(objective_values.keys()), list(objective_values.values()))
    #plt.show()
    print("Objective values: ", objective_values)


def run_only_once():
    optimize()
    #get_feasible_variables()
    
    arcs, path, picked_up = visualize()
    return arcs, picked_up

def run_pareto():
    create_pareto_front()

def find_extra_travel_time(picked_up):
    extra_time_per_rider = {}
    for driver in picked_up:
        for passenger in picked_up[driver]:
            pick_up_time = t[driver, passenger].x
            end_time = t[driver, passenger + nr_passengers].x
            shortest_path = T_ij[passenger, passenger + nr_passengers]
            extra_time = (end_time - pick_up_time) - shortest_path
            extra_time_per_rider[passenger] = extra_time
        destination_time = t[driver, driver_destination_nodes[driver]].x
        origin_time = t[driver, driver_origin_nodes[driver]].x
        direct_arc_time = T_ij[driver_origin_nodes[driver], driver_destination_nodes[driver]]
        extra_time_per_rider[driver] = (destination_time - origin_time) - direct_arc_time

    return extra_time_per_rider

#run_pareto()



arcs, picked_up = run_only_once()
runtime = time.time() - start_time
print('Total runtime: ', runtime)
print('Optimality gap: ', model.MIPGap)
print('Total number of passengers', nr_passengers)
print('Number of picked up passengers: ', model.objVal)
print('Picked up riders: ', picked_up)

print("Best bound: ", model.ObjBound)

extra_time_per_rider = find_extra_travel_time(picked_up)
print('Extra ride time per rider: ', extra_time_per_rider)

wb = Workbook()

sheet_1 = wb.add_sheet('Sheet 1')
sheet_1.write(0,1, 'Kjøretid')
sheet_1.write(0,2, 'Optimalitetsgap')
sheet_1.write(0,3, 'Passasjerer hentet')
sheet_1.write(0,4, 'Ekstra reisetid')
sheet_1.write(0,5, 'Rute')

sheet_1.write(1,1, runtime)
sheet_1.write(1,2, model.MIPGap)
sheet_1.write(1,3, model.objVal)
i = 2


total_extra_driver_time = 0
total_shortest_path_driver = 0
number_of_drivers = 0
total_extra_passenger_time = 0
total_shortest_path_passenger = 0
number_of_passengers = 0
max_driver_extra = 0
min_driver_extra = 0
max_passenger_extra = 0
min_passenger_extra = 0
extra_percentage_per_driver = 0
extra_percentage_per_passenger = 0


for rider in extra_time_per_rider:
    if rider in D:
        total_extra_driver_time += extra_time_per_rider[rider]
        extra_percentage_per_driver += extra_time_per_rider[rider] / T_k[rider]
        total_shortest_path_driver += T_ij[rider, driver_destination_nodes[rider]]
        if extra_time_per_rider[rider] >= max_driver_extra:
            max_driver_extra = extra_time_per_rider[rider]
        if extra_time_per_rider[rider] <= min_driver_extra:
            min_driver_extra = extra_time_per_rider[rider]
        number_of_drivers += 1
    else: 
        total_extra_passenger_time += extra_time_per_rider[rider]
        extra_percentage_per_passenger += extra_time_per_rider[rider] / T_k[rider]
        total_shortest_path_passenger += T_ij[rider, rider + nr_passengers]
        if extra_time_per_rider[rider] >= max_passenger_extra:
            max_passenger_extra = extra_time_per_rider[rider]
        if extra_time_per_rider[rider] <= min_passenger_extra:
            min_passenger_extra = extra_time_per_rider[rider]
        number_of_passengers += 1
    
average_extra_driver_time = total_extra_driver_time / number_of_drivers
average_extra_passenger_time = total_extra_passenger_time / number_of_passengers
average_shortest_path_driver = total_shortest_path_driver / number_of_drivers
average_shortest_path_passenger = total_shortest_path_passenger / number_of_passengers

sheet_1.write(1 ,7, average_extra_driver_time)
sheet_1.write(0, 7, 'Average extra driver time')
sheet_1.write(1, 6, average_extra_passenger_time)
sheet_1.write(0, 6, 'Average extra passenger time')

sheet_1.write(1 ,9, extra_percentage_per_driver / number_of_drivers) 
sheet_1.write(0, 9, 'Average extra driver time in %')
sheet_1.write(1, 8, extra_percentage_per_passenger / number_of_passengers)
sheet_1.write(0, 8, 'Average extra passenger time in %')

sheet_1.write(1 ,12, min_driver_extra)
sheet_1.write(0, 12, 'Minimum extra driver time')
sheet_1.write(1, 10, min_passenger_extra)
sheet_1.write(0, 10, 'Minimum extra passenger time')

sheet_1.write(1 ,13, max_driver_extra)
sheet_1.write(0, 13, 'Maximum extra driver time')
sheet_1.write(1, 11, max_passenger_extra)
sheet_1.write(0, 11, 'Maximum extra passenger time')


sheet_1.write(1,5, str(arcs))

sheet_1.write(0, 14, "Best bound")
sheet_1.write(1, 14, model.ObjBound)

wb.save(file_to_save)

print(result_solution)
print(result_bound)
print(result_time)

"""result_time.insert(0,0)

if model.objVal == result_bound[-1]:
    result_solution.append(result_bound[-1])
    result_time.append(model.Runtime)


if model.objVal != result_bound[-1]:
    print("Hei")
    last_solution = result_solution[-1]
    for i in range(int(result_time[-1]), 3600):
        print(i)
        result_solution.append(last_solution)
    for i in range(int(result_time[-1]), 3601):
        result_time.append(i)

def get_graph():
    plt.figure(2)
    plt.xlim(-160, 3800)
    plt.ylim(-1, result_bound[-1] + 1)
    plt.axhline(y = result_bound[0], color = 'r', linestyle = 'dashed', label = "UB = " + str(result_bound[-1]))
    plt.axvline(x = 3600, color = 'y', linestyle = 'dashed', label = "Time = 3600s" )

    plt.plot(result_time, result_solution)
    plt.ylabel('Passengers served')
    plt.xlabel('Time [s]')
    plt.legend()
    
    plt.legend(bbox_to_anchor = (0.85, 1.2), loc = 'upper center')
    
    plt.show()

get_graph()"""


#debug()


