import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# Load data from files and put maps into a dictionary labelled by sample angle.
steps = [20, 40, 100, 135]
icons = ["x", "^", "*", "."]

for index, step in enumerate(steps):
    file_name = f"../../Forming test data/Data/Strain data_All stages/Surfalex_20mm_001/Section one/major strain/Section1_{step:04d}.csv"
    first_frame = np.genfromtxt(file_name, delimiter=";", skip_header=6)
    with open(file_name, 'r') as input_file:
        first_time = float(input_file.readlines()[3].split(";")[2])

    file_name = f"../../Forming test data/Data/Strain data_All stages/Surfalex_10mm_002/Section one/major strain/Section1_{step + 1:04d}.csv"
    second_frame = np.genfromtxt(file_name, delimiter=";", skip_header=6)
    with open(file_name, 'r') as input_file:
        second__time = float(input_file.readlines()[3].split(";")[2])
    frame_time = second__time - first_time
    velocity = (second_frame[:, 3] - first_frame[:, 3]) / frame_time
    plt.plot(velocity, icons[index], label=step)

plt.xlabel("Sample Z position")
plt.ylabel("Velocity in z-direction (mm/s)")
plt.legend()
plt.show()
#plt.savefig(f"../figures/strain_maps.pdf", bbox_inches="tight")
