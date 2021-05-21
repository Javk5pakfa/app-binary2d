import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import savgol_filter


def smooth(x):
    return savgol_filter(x, 101, 3)


def td(x):
    return x[-1] - x[0]


def parse_filenames():
    num_file = 0
    file_names = []

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='*')
    args = parser.parse_args()

    for name in args.filename:
        file_names.append(name)
        num_file += 1

    return num_file, file_names


def load_raw_data():
    data = []
    num_files, file_names = parse_filenames()

    # Opens the pickle file, which is a dictionary of data.
    if num_files > 0:
        for name in file_names:
            with (open(name, "rb")) as openfile:
                while True:
                    try:
                        data.append(pickle.load(openfile))
                    except EOFError:
                        break
        # print(data[0]['config']['physics']['binary_eccentricity'])
    else:
        print("No file loaded!")

    return data


def get_orbital_evolution_data():
    e_c = []
    dedm = []

    for item in loaded_data:
        # Time component extrapolation.
        skip = 1
        eccentricity = item['config']['physics']['binary_eccentricity']
        i = np.where(item['time'] / 2.0 / np.pi < 200)[0][-1]

        # Binary data extrapolation.
        m1 = item['orbital_elements_change']['sink1'][:, 1][i::skip]
        m2 = item['orbital_elements_change']['sink2'][:, 1][i::skip]
        m = m1 + m2

        e_sink_1 = item['orbital_elements_change']['sink1'][:, 3][i::skip]
        e_sink_2 = item['orbital_elements_change']['sink2'][:, 3][i::skip]
        e_grav_1 = item['orbital_elements_change']['grav1'][:, 3][i::skip]
        e_grav_2 = item['orbital_elements_change']['grav2'][:, 3][i::skip]
        e_sink = e_sink_1 + e_sink_2
        e_grav = e_grav_1 + e_grav_2
        e_tot = e_sink + e_grav

        mean_edot = td(e_tot)
        mean_mdot = td(m)

        e_c.append(eccentricity)
        dedm.append(mean_edot / mean_mdot)

    return e_c, dedm


if __name__ == "__main__":
    loaded_data = load_raw_data()
    eccentricities, dedms = get_orbital_evolution_data()

    print(eccentricities)
    print(dedms)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.scatter(eccentricities, dedms)
    plt.suptitle('Binary orbital evolution, MN = 10')
    plt.xlabel('e')
    plt.ylabel(r'$\frac{\Delta e}{\Delta M}$')
    plt.show()
