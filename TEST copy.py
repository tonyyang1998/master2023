import matplotlib.pyplot as plt

path = {
    0: [
        (4.993453, 60.388651),
        (5.087813, 60.352223),
        (5.039608, 60.390343),
        (5.265, 60.379),
        (5.152274, 60.354616),
        (5.326163, 60.395),
    ],
    1: [
        (5.993453, 61.388651),
        (6.087813, 61.352223),
        (6.039608, 61.390343),
        (6.265, 61.379),
        (6.152274, 61.354616),
        (6.326163, 61.395),
    ]
}

# Define colors for different node types
colors = {'start': 'green', 'pickup': 'blue', 'delivery': 'red', 'end': 'yellow'}

def plot_path(path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all nodes for each driver
    for driver, nodes in path.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            node_type = ''
            if i == 0:
                node_type = 'start'
            elif i == n - 1:
                node_type = 'end'
            elif i < n/2:
                node_type = 'pickup'
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

plot_path(path)
