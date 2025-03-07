import numpy as np
from SurvivalEVAL.Evaluations.util import predict_multi_probs_from_curve

def estimate_concordance_index(
        event_indicator: np.ndarray,
        event_time: np.ndarray,
        estimate: np.ndarray,
        bg_event_time: np.ndarray = None,
        partial_weights: np.ndarray = None,
        tied_tol: float = 1e-8
):
    order = np.argsort(event_time, kind="stable")

    comparable, tied_time, weight = get_comparable(event_indicator, event_time, order)

    if partial_weights is not None:
        event_indicator = np.ones_like(event_indicator)
        comparable_2, tied_time, weight = get_comparable(event_indicator, bg_event_time, order, partial_weights)
        for ind, mask in comparable.items():
            weight[ind][mask] = 1
        comparable = comparable_2

    if len(comparable) == 0:
        raise ValueError("Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        # w_i = partial_weights[order[ind]] # change this
        w_i = weight[ind]
        weight_i = w_i[order[mask]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        # n_ties = ties.sum()
        n_ties = np.dot(weight_i, ties.T)
        # an event should have a higher score
        con = est < est_i
        # n_con = con[~ties].sum()
        con[ties] = False
        n_con = np.dot(weight_i, con.T)

        # numerator += w_i * n_con + 0.5 * w_i * n_ties
        # denominator += w_i * mask.sum()
        numerator += n_con + 0.5 * n_ties
        denominator += np.dot(w_i, mask.T)

        tied_risk += n_ties
        concordant += n_con
        # discordant += est.size - n_con - n_ties
        discordant += np.dot(w_i, mask.T) - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time

def get_comparable(event_indicator: np.ndarray, event_time: np.ndarray, order: np.ndarray,
                   partial_weights: np.ndarray = None):
    if partial_weights is None:
        partial_weights = np.ones_like(event_indicator, dtype=float)
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    weight = {}

    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        end = i + 1
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time

        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
                weight[j] = partial_weights[order] * partial_weights[order[j]]
        i = end

    return comparable, tied_time, weight

def predict_multi_probabilities_from_curve(predicted_curves, time_coordinates, target_times, interpolation) -> np.ndarray:
    """
    Predict the probability of event at multiple time points from the predicted curve.
    param target_times: array-like, shape = (n_target_times)
        Time points at which the probability of event is to be predicted.
    :return: array-like, shape = (n_samples, n_target_times)
        Predicted probabilities of event at the target time points.
    """
    predict_probs_mat = []
    for i in range(predicted_curves.shape[0]):
        predict_probs = predict_multi_probs_from_curve(predicted_curves[i, :], time_coordinates,
                                                       target_times, interpolation).tolist()
        predict_probs_mat.append(predict_probs)
    predict_probs_mat = np.array(predict_probs_mat)
    return predict_probs_mat