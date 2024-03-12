from Cycles import GraphCreator, MemoryZone
from scheduling import create_initial_sequence, create_starting_config, run_simulation

# parser = argparse.ArgumentParser()
# parser.add_argument("config_file", help="path to json config file")
# # parser.add_argument("--plot", action="store_true", help="plot grid")
# args = parser.parse_args()

# with pathlib.Path(args.config_file).open("r") as f:
#     config = json.load(f)
# arch = config["arch"]
# max_timesteps = config["max_timesteps"]
# num_ion_chains = config["num_ion_chains"]
# filename = config["qu_alg"]

archs = [[12, 12, 3, 3]]  # [4, 4, 2, 2], [7, 7, 1, 1], [10, 10, 1, 1], [3, 3, 5, 5]]
# arch = [10, 10, 2, 2]
# arch = [4, 4, 2, 2]
for arch in archs:
    filename = "QASM_files/qft_62qubits.qasm"
    num_ion_chains = 62
    max_timesteps = 100000

    seed = 0
    m, n, v, h = arch
    # create dummy graph
    graph = GraphCreator(m, n, v, h).get_graph()
    n_of_traps = len([trap for trap in graph.edges() if graph.get_edge_data(trap[0], trap[1])["edge_type"] == "trap"])
    ion_chains, number_of_registers = create_starting_config(num_ion_chains, graph, seed=seed)

    print(f"arch: {arch}, seed: {seed}, registers: {number_of_registers}\n")

    time_2qubit_gate = 3
    time_1qubit_gate = 1
    max_chains_in_parking = 3

    memorygrid = MemoryZone(
        m,
        n,
        v,
        h,
        ion_chains,
        max_timesteps,
        max_chains_in_parking,
        time_2qubit_gate=time_2qubit_gate,
        time_1qubit_gate=time_1qubit_gate,
    )

    memorygrid.update_distance_map()
    seq, flat_seq, dag_dep, next_node_initial = create_initial_sequence(memorygrid.distance_map, filename)
    run_simulation(memorygrid, max_timesteps, seq, flat_seq, dag_dep, next_node_initial, max_length=10, show_plot=False)
