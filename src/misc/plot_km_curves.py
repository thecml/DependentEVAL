from matplotlib import pyplot as plt
import numpy as np

class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    def __init__(self, event_times, event_indicators):
        self.event_times = event_times
        self.event_indicators = event_indicators

        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities

def compare_km_curves(df1, df2, intervals=None, save_fig=None, show=False):
    """
    Plots the KM curves of the original and modified dataframe
    :param df1: original dataframe
    :param df2: combined truth and censored data
    :return: None
    """
    #results = logrank_test(df1.time.values, df2.time.values, df1.event.values, df2.event.values)

    event_times_1 = df1.time.values[df1.event.values == 1]
    censor_times_1 = df1.time.values[df1.event.values == 0]
    event_times_2 = df2.time.values[df2.event.values == 1]
    censor_times_2 = df2.time.values[df2.event.values == 0]
    if intervals is None:
        intervals = 21  # 20 bins
    bins = np.linspace(0, round(df1.time.max()), intervals)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax0.hist([event_times_1, censor_times_1], bins=bins, histtype='bar', stacked=True)
    ax0.legend(['Event times', 'Censor Times'])
    ax0.set_title("Event/Censor Time Histogram")

    km_estimator = KaplanMeier(df1.time.values, df1.event.values)
    ax1.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    ax1.set_title("Kaplan-Meier Curve")
    ax1.set_ylim([0, 1])
    xmin, xmax = ax1.get_xlim()

    ax2.hist([event_times_2, censor_times_2], bins=bins, histtype='bar', stacked=True)
    ax2.legend(['Event times', 'Censor Times'])
    # ax2.set_title("Event/Censor Times Histogram")

    km_estimator = KaplanMeier(df2.time.values, df2.event.values)
    ax3.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    # ax3.set_title("Kaplan-Meier Curve")
    ax3.set_ylim([0, 1])
    ax3.set_xlim([xmin, xmax])

    # fig.set_size_inches(12, 12)
    #plt.suptitle('Logrank Test: p-value = {:.5f}'.format(results.p_value))
    plt.setp(ax0, xlabel='Time', ylabel='Counts')
    plt.setp(ax1, xlabel='Time', ylabel='Probabilities')
    plt.setp(ax2, xlabel='Time', ylabel='Counts')
    plt.setp(ax3, xlabel='Time', ylabel='Probabilities')
    # plt.show()
    if save_fig is not None:
        fig.savefig(save_fig, dpi=300)
        
    if show:
        plt.show()
        