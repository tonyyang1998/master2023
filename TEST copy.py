import matplotlib.pyplot as plt

path = {
    0: [
        (4.790867, 60.661021),
        (4.892748, 60.538108),
        (5.039608, 60.390343),
        (5.265, 60.379),
        (5.326163, 60.395),
        (5.352, 60.3635),
    ],
    1: [(5.135965, 60.18781),
        (4.982022, 60.261564),
        (5.068799, 60.25957),
        (5.259, 60.291),
    ]
}

nodes_visited = {
    0: [(0, 0),
        (2, 2),
        (3, 4),
        (5, 2),
        (6, 2),
        (8, 0),
    ],
    1: [(1, 0),
        (4, 1),
        (7, 2),
        (9, 0),
    ]
}

# Define colors for different node types
colors = {'start': 'red', 'pickup': 'blue', 'delivery': 'turquoise', 'end': 'limegreen'}

def plot_path(path, nodes_visited):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Map pickup coordinates to passenger numbers
    pickup_coords = {}
    for driver, pickup_list in nodes_visited.items():
        for pickup in pickup_list:
            passenger_num = pickup[0]
            pickup_index = pickup_list.index(pickup)
            pickup_coord = path[driver][pickup_index]
            pickup_coords[pickup_coord] = f"P{passenger_num}"

    # Plot all nodes for each driver
    for driver, nodes in path.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            node_type = ''
            if i == 0:
                node_type = 'start'
                plt.annotate(f"DS{driver}", (node[0], node[1]))
            elif i == n - 1:
                node_type = 'end'
                plt.annotate(f"DE{driver}", (node[0], node[1]))
            elif i < n/2:
                node_type = 'pickup'
                if node in pickup_coords:
                    plt.annotate(pickup_coords[node], (node[0], node[1]))
            else:
                node_type = 'delivery'
            plt.scatter(node[0], node[1], color=colors[node_type])

        # Plot the paths between the nodes for each driver
        for i in range(len(nodes) - 1):
            plt.plot([nodes[i][0], nodes[i+1][0]], [nodes[i][1], nodes[i+1][1]], color='black')

    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Driver Paths")
    
    plt.show()


plot_path(path, nodes_visited)


