from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np


class Metrics:
    #     trainer.plotter.estimator.q_99
    def __init__(self, params, GT, alpha):
        self.alpha = alpha
        self.params = params

        A = GT[0]
        self.tp_zones, self.fp_zones = [], []

        tp_half_width = params.W

        for mid in A:
            self.tp_zones.append((int(mid - tp_half_width), int(mid + tp_half_width)))

        for i in range(len(self.tp_zones)):
            if i == 0:
                self.fp_zones.append((0, self.tp_zones[i][0]))
            elif i == len(self.tp_zones) - 1:
                self.fp_zones.append((self.tp_zones[i][1], params.Ts - 1))
            else:
                self.fp_zones.append((self.tp_zones[i - 1][1], self.tp_zones[i][0]))

    def get(self, proj):
        peaks, _ = find_peaks(proj, height=self.alpha, distance=self.params.W)

        TPZ = np.zeros((len(self.tp_zones),))
        FPZ = np.zeros((len(self.fp_zones),))

        for p in peaks:
            for i, (st, en) in enumerate(self.tp_zones):
                if (st <= p) and (p < en):
                    TPZ[i] = 1.0
            for i, (st, en) in enumerate(self.fp_zones):
                if (st <= p) and (p < en):
                    FPZ[i] = 1.0

        P = len(TPZ)  # condition positive
        N = len(FPZ)  # condition negative

        true_positives = sum(TPZ == 1)
        false_positives = sum(FPZ == 1)
        false_negatives = sum(TPZ == 0)
        true_negatives = sum(FPZ == 0)

        tpr = true_positives / P
        fpr = false_positives / N
        fnr = false_negatives / P
        tnr = true_negatives / N

        assert tpr + fnr == 1, 'check metrics.get'
        assert fpr + tnr == 1, 'check metrics.get'

        metrics_dict = dict(
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
        )
        return metrics_dict


def get_metrics(out, A, q_99, params, debug=False, X_=None):
    """ 
    out:
        (np.array) convolution of the data with the filter
    A:
        list of midddles of sequences embedded in the data, time stamps
    q_99:
        99th quantile of the null dist
    returns:
        (int) TP, FP, FN
    """

    peaks, _ = find_peaks(out, height=q_99, distance=params.W)

    I = []
    for i, p in enumerate(peaks):
        if out[p] > q_99:  # if the height of that peak is high
            I.append(i)
    peaks = peaks[I]

    TP, FP, FN = 0, 0, 0

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(18, 4))
        axt = ax.twinx()
        ax.spy(X_, aspect='auto', origin='lower', markersize=0.7, color='k')
        axt.plot(out)
        axt.axhline(q_99, lw=3, ls=':', color='red')
        axt.plot(peaks, out[peaks], 'r*', markersize=30)
        for mid in A:
            axt.axvline(mid, lw=1, color='red')
        ax.set_xlim(0, 4000)
        axt.set_ylim(0, 3 * q_99)

    for p in peaks:  # p are peak indexes
        distToClosestRealSeq = min(abs(p - np.array(A)))  # distance to closess seq_mid from current peak
        if distToClosestRealSeq < params.W:  # if dist is small enough, discovery
            TP += 1
        else:
            FP += 1

    if not len(peaks) == 0:
        for mid in A:  # mid are seq_mids
            distToClosestRealSeq = min(abs(mid - peaks))  # distance to closess peak from current seq_mid
            closestPeakIdx = np.argmin(abs(mid - peaks))
            if distToClosestRealSeq > params.W:  # if that distance is large, non-discovery
                FN += 1
                if debug:
                    plt.axvline(peaks[closestPeakIdx], lw=2, color='green')
            else:  # else check if the peak is significant
                if out[peaks[closestPeakIdx]] < q_99:
                    FN += 1
                    if debug:
                        plt.axvline(peaks[closestPeakIdx], lw=2, color='green')
    else:
        return 0, 0, len(A)  # if no significan peaks exist, return all non-discovered

    return TP, FP, FN


class PPSeqMetrics:

    def __init__(self, GT, Ts, W):

        A = GT[0]
        self.tp_zones, self.fp_zones = [], []

        tp_half_width = W

        for mid in A:
            self.tp_zones.append((int(mid - tp_half_width), int(mid + tp_half_width)))

        for i in range(len(self.tp_zones)):
            if i == 0:
                self.fp_zones.append((0, self.tp_zones[i][0]))
            elif i == len(self.tp_zones) - 1:
                self.fp_zones.append((self.tp_zones[i][1], Ts - 1))
            else:
                self.fp_zones.append((self.tp_zones[i - 1][1], self.tp_zones[i][0]))

    def get(self, peaks):

        TPZ = np.zeros((len(self.tp_zones),))
        FPZ = np.zeros((len(self.fp_zones),))

        for p in peaks:
            for i, (st, en) in enumerate(self.tp_zones):
                if (st <= p) and (p < en):
                    TPZ[i] = 1.0
            for i, (st, en) in enumerate(self.fp_zones):
                if (st <= p) and (p < en):
                    FPZ[i] = 1.0

        P = len(TPZ)  # condition positive
        N = len(FPZ)  # condition negative

        true_positives = sum(TPZ == 1)
        false_positives = sum(FPZ == 1)
        false_negatives = sum(TPZ == 0)
        true_negatives = sum(FPZ == 0)

        tpr = true_positives / P
        fpr = false_positives / N
        fnr = false_negatives / P
        tnr = true_negatives / N

        assert tpr + fnr == 1, 'check metrics.get'
        assert fpr + tnr == 1, 'check metrics.get'

        metrics_dict = dict(
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
        )
        return metrics_dict