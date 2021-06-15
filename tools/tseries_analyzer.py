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
    """
    This function parses file paths provided on the command line and store them
    for use.
    """

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
    else:
        print("No file loaded!")

    return data


def get_a_dedm_data(data):
    a_c = []
    dedm = []

    for item in data:
        # Time component extrapolation.
        skip = 1
        eccentricity = item['config']['physics']['binary_eccentricity']
        i = np.where(item['time'] / 2.0 / np.pi < 200)[0][-1]

        # Binary data extrapolation.
        m1 = item['orbital_elements_change']['sink1'][:, 1][i::skip]
        m2 = item['orbital_elements_change']['sink2'][:, 1][i::skip]
        m = m1 + m2

        a_sink_1 = item['orbital_elements_change']['sink1'][:, 0][i::skip]
        a_sink_2 = item['orbital_elements_change']['sink2'][:, 0][i::skip]
        a_grav_1 = item['orbital_elements_change']['grav1'][:, 0][i::skip]
        a_grav_2 = item['orbital_elements_change']['grav2'][:, 0][i::skip]
        a_sink = a_sink_1 + a_sink_2
        a_grav = a_grav_1 + a_grav_2
        a_tot = a_sink + a_grav

        mean_adot = td(a_tot)
        mean_mdot = td(m)

        a_c.append(eccentricity)
        dedm.append(mean_adot / mean_mdot)

    return a_c, dedm


def get_ecc_dedm_data(data):
    """
    This function uses input pickle checkpoint files to get the eccentricity
    associated with each as well as the mean eccentricity change per mass
    accreted. The checkpoint files analyzed should contain time series data
    of runs longer than 200 orbits, or the time it takes the system to settle.
    They should also have identical parameters other than eccentricity.

    :return: eccentricity [] and de/dm []
    """

    e_c = []
    dedm = []

    for item in data:
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


def get_orbit_ydata(data, index):
    orbit_list = []
    ydata_list = []

    for item in data:
        # Time component extrapolation.
        skip = 1
        i = np.where(item['time'] / 2.0 / np.pi < 200)[0][-1]
        t = item['time'][i::skip]
        orbit = t / 2.0 / np.pi
        orbit_list.append(orbit[1:])

        # Binary data extrapolation.
        m1 = item['orbital_elements_change']['sink1'][:, 1][i::skip]
        m2 = item['orbital_elements_change']['sink2'][:, 1][i::skip]
        m = m1 + m2
        mean_mdot = td(m) / td(t)
        # mdot = smooth(np.diff(m) / np.diff(t)) / mean_mdot

        sink_1 = item['orbital_elements_change']['sink1'][:, index][i::skip]
        sink_2 = item['orbital_elements_change']['sink2'][:, index][i::skip]
        grav_1 = item['orbital_elements_change']['grav1'][:, index][i::skip]
        grav_2 = item['orbital_elements_change']['grav2'][:, index][i::skip]
        sink = sink_1 + sink_2
        grav = grav_1 + grav_2
        tot = sink + grav

        # edot_acc = smooth(np.diff(e_sink) / np.diff(t)) / mean_mdot
        # edot_grac = smooth(np.diff(e_grav) / np.diff(t)) / mean_mdot
        dot_acc = smooth(np.diff(tot) / np.diff(t)) / mean_mdot

        ydata_list.append(dot_acc)

    return orbit_list, ydata_list


def plot_xy(x_data, y_data, x_label, y_label, super_title):
    """
    This function is a generic plotting function.

    :param x_data: []
    :param y_data: []
    :param x_label: str
    :param y_label: str
    :param super_title: str
    :return: Plot of data x vs y in linear scale.
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.scatter(x_data, y_data)
    plt.suptitle(super_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    title = 'title'
    # xlabel = 'e'
    xlabel = 'orbit'
    # ylabel = r'$\frac{\Delta e}{\Delta M}$'
    ylabel = r'$\Delta e$'
    loaded_data = load_raw_data()
    orbits, ydata = get_orbit_ydata(loaded_data)
    # eccentricities, dedms = get_ecc_dedm_data(loaded_data)
    plot_xy(orbits, ydata, xlabel, ylabel, title)
    # plot_xy(eccentricities, dedms, xlabel, ylabel, title)
