from graph_utils import GraphCreator, PZCreator, ProcessingZone, create_idc_dictionary
from Cycles import create_starting_config, find_path_edge_to_edge
from scheduling import get_ion_chains
from shuttle import main
from compilation import compile
import math
import networkx as nx
import numpy as np
from datetime import datetime
from plotting import plot_state
from graph_utils import get_idx_from_idc

plot = False
save = False

paths = False
cycle_or_paths = "Paths" if paths else "Cycles"

failing_junctions = 0

number_of_pzs_list = [2]  # [2, 3, 4]#, 5, 6, 7, 8, 9, 10]
archs = [
    [5, 5, 1, 1],
]
seeds = [2]  # , 1, 2, 3, 4]
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for m, n, v, h in archs:
    for number_of_pzs in number_of_pzs_list:
        timesteps_array = []
        cpu_time_array = []

        for seed in seeds:
            start_time = datetime.now()

            exit1 = (float(m-1), float(n-1))
            entry1 = (float(m-1), float(0))
            processing_zone1 = (float(m), float(n))

            exit2 = (0.0, 0.0)
            entry2 = (0.0, float(n-1))
            processing_zone2 = (-1.0, -1.0)


            pz1 = ProcessingZone("pz1", [exit1, entry1, processing_zone1])
            pz2 = ProcessingZone("pz2", [exit2, entry2, processing_zone2])
            pzs = [pz1, pz2]
            basegraph_creator = GraphCreator(m, n, v, h, failing_junctions, pzs)
            MZ_graph = basegraph_creator.get_graph()        
            pzgraph_creator = PZCreator(m, n, v, h, failing_junctions, pzs)
            G = pzgraph_creator.get_graph()
            G.mz_graph = MZ_graph

            G.idc_dict = create_idc_dictionary(G)
            G.idc_dict = create_idc_dictionary(G)
            G.pzs = pzs
            G.parking_edges_idxs = []
            for pz in G.pzs:
                G.parking_edges_idxs.append(get_idx_from_idc(G.idc_dict, pz.parking_edge))
            print(f"parking_edges_idxs: {G.parking_edges_idxs}")
            
            G.max_num_parking = 3
            for pz in G.pzs:
                pz.max_num_parking = G.max_num_parking # if changed here, also change in shuttle.py (check_duplicates) and check for further updates to max_num_parking

            G.plot = plot
            G.save = save
            G.arch = str([m, n, v, h])

            number_of_chains = math.ceil(len(MZ_graph.edges()) / 2)
            

            # plot for paper
            # plot_state(
            #     G, (None, None), plot_ions=True, show_plot=plot, save_plot=save
            # )

            print(f"Number of chains: {number_of_chains}")
            #algorithm = "qft_no_swaps_nativegates_quantinuum_tket"
            algorithm = "full_register_access"
            qasm_file_path = (
                f"../../../QASM_files/{algorithm}/{algorithm}_{number_of_chains}.qasm"
            )

            edges = list(G.edges())

            create_starting_config(G, number_of_chains, seed=seed)
            G.idc_dict = create_idc_dictionary(G)
            G.state = get_ion_chains(G)

            sequence = compile(qasm_file_path)
            G.sequence = sequence
            print(len(sequence), "len seq")

            # if there is a real tuple in sequence (2-qbuit gate) use partitioning
            # if any(len(i) > 1 for i in sequence):
            #     part = get_partition(qasm_file_path, len(G.pzs))
            #     partition = {pz.name: part[i] for i, pz in enumerate(G.pzs)}
            #     num_pzs = len(G.pzs)
            # else:
            if True:
                # else place them in the closest processing zone (equally distributed)
                # TODO double check
                partition = {pz.name: [] for pz in G.pzs}
                # Assign each ion to the closest processing zone
                for ion, position in G.state.items():
                    closest_pz = None
                    min_distance = float("inf")
                    for pz in G.pzs:
                        distance = len(find_path_edge_to_edge(G, position, pz.parking_edge))
                        if distance < min_distance:
                            min_distance = distance
                            closest_pz = pz
                    partition[closest_pz.name].append(ion)
                # Balance the ions among the processing zones
                all_ions = [ion for ions in partition.values() for ion in ions]
                all_ions.sort(key=lambda ion: G.state[ion])
                num_pzs = len(G.pzs)
                ions_per_pz = len(all_ions) // num_pzs
                for i, pz in enumerate(G.pzs):
                    start_index = i * ions_per_pz
                    end_index = start_index + ions_per_pz
                    partition[pz.name] = all_ions[start_index:end_index]
                # Distribute any remaining ions
                remaining_ions = all_ions[num_pzs * ions_per_pz :]
                for i, ion in enumerate(remaining_ions):
                    partition[G.pzs[i % num_pzs].name].append(ion)

            print('partition: ', partition)

            # Create a reverse mapping from element to partition name
            map_to_pz = {
                element: pz
                for pz, elements in partition.items()
                for element in elements
            }
            G.map_to_pz = map_to_pz

            # Ensure all elements are in one of the partitions
            all_partition_elements = []
            for elements in partition.values():
                all_partition_elements.extend(elements)
            unique_sequence = []
            for seq_elem in sequence:
                for elem in seq_elem:
                    if elem not in unique_sequence:
                        unique_sequence.append(elem)
            assert all(element in all_partition_elements for element in unique_sequence)

            if len(G.pzs) > 1:
                # and no element is in both partitions
                pz_sets = {pz: set(elements) for pz, elements in partition.items()}
                common_elements = set.intersection(*pz_sets.values())
                assert (
                    not common_elements
                ), f"{common_elements} are overlapping in partitions"

            timesteps = main(G, sequence, partition, cycle_or_paths)
            end_time = datetime.now()
            cpu_time = end_time - start_time

            timesteps_array.append(timesteps)
            cpu_time_array.append(cpu_time)

            # save timesteps in a file
            with open(f"benchmarks/{time}{algorithm}.txt", "a") as f:
                f.write(
                    f"{m, n, v, h}, #pzs: {num_pzs}, ts: {timesteps}, seed: {seed}\n"
                )

        # calculate averages
        timesteps_array = np.array(timesteps_array)
        cpu_time_array = np.array(cpu_time_array)
        timesteps_average = np.mean(timesteps_array)
        cpu_time_average = np.mean(cpu_time_array)

        # save averages
        with open(f"benchmarks/{time}{algorithm}.txt", "a") as f:
            f.write(
                f"{m, n, v, h}, #pzs: {num_pzs}, average_ts: {timesteps_average}, average_cpu_time: {cpu_time_average}\n"
            )


# if __name__ == "__main__":
#     gate_info_list = {'pz1': [1], 'pz2': [2, 1], 'pz3': [3], 'pz4': [], 'pz5': []}
#     sequence = [(2, 1), (1,), (3,)]
#     print(find_pz_order(G, sequence, gate_info_list))


# TODOs:
# - TODO 1: gate_execution_finished -> implement different gate execution times
# - TODO 2: other_next_edges -> needed for path out of pz if cycles is True - would now have to be calculated for each pz before
# - TODO 3: new_gate_starting -> would need to know next gate at pz - checks in scheduling.py if new gate can start with ions in parking - would stop all ions in exit