import glob
from typing import List, Dict, Tuple

from dic_analysis import DeformationMap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_data(sample_angles: List[str]) -> Dict[str, List[DeformationMap]]:
    """Load data from files and put maps into a dictionary labelled by sample angle."""
    deformation_maps = {}
    for angle in tqdm(sample_angles):
        data_folder = f"../../DIC Data/Data/test {angle}/displacement data/"
        file_list = glob.glob(f"{data_folder}*")
        deformation_maps[angle] = [DeformationMap(file_path, [0, 1, 2, 3]) for
                                   file_path in file_list]
    return deformation_maps


def calculate_mean_strain(sample_angles: List[str],
                          deformation_maps: Dict[str, List[DeformationMap]],
                          x_range: Tuple[int, int],
                          y_range: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray],
                                                             Dict[str, np.ndarray]]:
    """Calculate mean strain and transverse strain over a region of the maps specified by
     x-range and y-range."""
    mean_strain = {}
    mean_trans_strain = {}
    # Loop over all sample angles
    for angle in sample_angles:
        mean_strain[angle] = []
        mean_trans_strain[angle] = []
        # Loop over all time steps except the first as the first is always zero.
        for def_map in deformation_maps[angle][1:]:
            # Crop the map to the center and calculate the mean longitudinal strain
            cropped_map = def_map.f22[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            mean_strain[angle].append(np.mean(cropped_map))
            # Crop the map to the center and calculate the mean transverse strain
            cropped_map = def_map.f11[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            mean_trans_strain[angle].append(np.mean(cropped_map))
        # Convert list of mean strains to np array
        mean_strain[angle] = np.array(mean_strain[angle])
        mean_trans_strain[angle] = np.array(mean_trans_strain[angle])
    return mean_strain, mean_trans_strain


def main():
    sample_angles = ["0-1", "0-2", "30-1", "30-2", "45-1", "45-2", "60-1", "60-2", "90-1", "90-2"]
    deformation_maps = load_data(sample_angles)

    # We crop the deformation map to only consider the center of the sample.
    # The strain is calculated over this region.
    x_range = (1, 12)
    y_range = (10, 24)

    # Calculation of mean strain over time, one for each sample angle
    mean_strain, mean_trans_strain = calculate_mean_strain(sample_angles, deformation_maps,
                                                           x_range, y_range)

    # We cut data at a min and max strain to avoid noisy data
    min_strain = 0.02
    max_strain = 0.29

    for angle in sample_angles:
        strain_ratio = - mean_trans_strain[angle] / mean_strain[angle]
        lankford = strain_ratio / (1 - strain_ratio)

        mask = np.logical_and(min_strain < mean_strain[angle], mean_strain[angle] < max_strain)
        plt.plot(mean_strain[angle][mask], lankford[mask], label=angle)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("strain")
    plt.ylabel("Lankford parameter")
    plt.xlim(0, 0.25)
    plt.savefig(f"../figures/lankford_parameter.pdf", bbox_inches='tight')


main()
