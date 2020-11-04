from dic_analysis import DeformationMap
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# Load data from files and put maps into a dictionary labelled by sample angle.
deformation_maps = {}

sample_angles = ["0-1", "45-1", "90-1"]
time_steps = [[50, 100, 200, 210, 222], [50, 100, 200, 210, 227], [50, 100, 200, 210, 228]]

num_angles = len(sample_angles)
num_steps = len(time_steps[0])

fig, axes = plt.subplots(nrows=num_angles, ncols=num_steps)

for index, ax in enumerate(axes.flat):
    angle = sample_angles[index // num_steps]
    step = time_steps[index // num_steps][index % num_steps]

    data_file = f"../../DIC Data/Data/test {angle}/displacement data/data_{step:03d}.csv"
    deformation_map = DeformationMap(data_file, [0, 1, 2, 3])

    im = ax.imshow(deformation_map.f22, cmap='viridis', interpolation='none', vmin=0, vmax=0.4,
                   aspect=0.5)
    ax.set_title(f"{step}")
    ax.set_xticks([])
    ax.set_yticks([])

fig.text(0.02, 0.9, "a)", size=20)
fig.text(0.02, 0.57, "b)", size=20)
fig.text(0.02, 0.25, "c)", size=20)


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.subplots_adjust(left=0.03, bottom=0.05, right=0.85, top=0.95, wspace=0.05, hspace=0.25)
plt.show()


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

fig.savefig(f"../figures/strain_maps.pdf", bbox_inches="tight")
