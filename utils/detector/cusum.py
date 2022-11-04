"""Cumulative sum algorithm (CUSUM) to detect abrupt changes in data."""
# https://github.com/demotu/detecta/blob/master/detecta/detect_cusum.py
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu'
__version__ = "1.0.5"
__license__ = "MIT"

"""Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.
   Parameters
   ----------
   x : 1D array_like
       data.
   threshold : positive number, optional (default = 1)
       amplitude threshold for the change in the data.
   drift : positive number, optional (default = 0)
       drift term that prevents any change in the absence of change.
   ending : bool, optional (default = False)
       True (1) to estimate when the change ends; False (0) otherwise.
   show : bool, optional (default = True)
       True (1) plots data in matplotlib figure, False (0) don't plot.
   ax : a matplotlib.axes.Axes instance, optional (default = None).
   Returns
   -------
   ta : 1D array_like [indi, indf], int
       alarm time (index of when the change was detected).
   tai : 1D array_like, int
       index of when the change started.
   taf : 1D array_like, int
       index of when the change ended (if `ending` is True).
   amp : 1D array_like, float
       amplitude of changes (if `ending` is True).
   Notes
   -----
   Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
   Start with a very large `threshold`.
   Choose `drift` to one half of the expected change, or adjust `drift` such
   that `g` = 0 more than 50% of the time.
   Then set the `threshold` so the required number of false alarms (this can
   be done automatically) or delay for detection is obtained.
   If faster detection is sought, try to decrease `drift`.
   If fewer false alarms are wanted, try to increase `drift`.
   If there is a subset of the change times that does not make sense,
   try to increase `drift`.
   Note that by default repeated sequential changes, i.e., changes that have
   the same beginning (`tai`) are not deleted because the changes were
   detected by the alarm (`ta`) at different instants. This is how the
   classical CUSUM algorithm operates.
   If you want to delete the repeated sequential changes and keep only the
   beginning of the first sequential change, set the parameter `ending` to
   True. In this case, the index of the ending of the change (`taf`) and the
   amplitude of the change (or of the total amplitude for a repeated
   sequential change) are calculated and only the first change of the repeated
   sequential changes is kept. In this case, it is likely that `ta`, `tai`,
   and `taf` will have less values than when `ending` was set to False.
   See this IPython Notebook [2]_.
   References
   ----------
   .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
   .. [2] https://github.com/demotu/detecta/blob/master/docs/detect_cusum.ipynb
   Examples
   --------
   # >>> x = np.random.randn(300)/5
   # >>> x[100:200] += np.arange(0, 4, 4/100)
   # >>> ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)
   # >>> x = np.random.randn(300)
   # >>> x[100:200] += 6
   # >>> detect_cusum(x, 4, 1.5, True, True)
   # >>> x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
   # >>> ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)

   Version history
   ---------------
   '1.0.5':
       Part of the detecta module - https://pypi.org/project/detecta/
   """


class CUSUM:
    def __init__(self, threshold=1.0, drift=0, ending=False, show=False, ax=None):
        self.X = None
        self.threshold = threshold
        self.drift = drift
        self.ending = ending
        self.show = show
        self.ax = ax
        self.ta = None
        self.tai = None
        self.taf = None
        self.gp = 0
        self.gn = 0

    def detect_cusum(self, s):  # s is residual
        self.gp = self.gp + s - self.drift
        self.gn = self.gn - s - self.drift
        if self.gp < 0:
            self.gp = 0
        if self.gn < 0:
            self.gn = 0
        if self.gp > self.threshold or self.gn > self.threshold:  # change detected!
            # ta = np.append(ta, i)  # alarm index
            # tai = np.append(tai, tap if gp[i] > self.threshold else tan)  # start
            self.gp, self.gn = 0, 0  # reset alarm
            return True
        else:
            return False
        # self.X = X
        #
        # for x in self.X:
        #     x = np.atleast_1d(x).astype('float64')
        #     gp, gn = np.zeros(x.size), np.zeros(x.size)
        #     ta, tai, taf = np.array([[], [], []], dtype=int)
        #     tap, tan = 0, 0
        #     amp = np.array([])
        #     # Find changes (online form)
        #     for i in range(1, x.size):
        #         s = x[i] - x[i - 1]
        #         gp[i] = gp[i - 1] + s - self.drift  # cumulative sum for + change
        #         gn[i] = gn[i - 1] - s - self.drift  # cumulative sum for - change
        #         if gp[i] < 0:
        #             gp[i], tap = 0, i
        #         if gn[i] < 0:
        #             gn[i], tan = 0, i
        #         if gp[i] > self.threshold or gn[i] > self.threshold:  # change detected!
        #             ta = np.append(ta, i)  # alarm index
        #             tai = np.append(tai, tap if gp[i] > self.threshold else tan)  # start
        #             gp[i], gn[i] = 0, 0  # reset alarm
        #
        #     # THE CLASSICAL CUSUM ALGORITHM ENDS HERE
        #     if self.show:
        #         _plot(x, self.threshold, self.drift, self.ending, self.ax, ta, tai, taf, gp, gn)
        #


def _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
    """Plot results of the detect_cusum function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
            if ending:
                ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                         label='Ending')
            ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                     label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax1.set_title('Time series and detected changes ' +
                      '(threshold= %.3g, drift= %.3g): N changes = %d'
                      % (threshold, drift, len(tai)))
        ax2.plot(t, gp, 'y-', label='+')
        ax2.plot(t, gn, 'm-', label='-')
        ax2.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
        ax2.set_xlabel('Data #', fontsize=14)
        ax2.set_ylim(-0.01 * threshold, 1.1 * threshold)
        ax2.axhline(threshold, color='r')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    x = np.random.rand(4, 300) / 5
    x[0][100:200] += np.arange(0, 4, 4 / 100)
    x[1][100:200] += np.arange(0, 4, 4 / 100)
    cusum = CUSUM(x, 1.5, .02, True, True)
    cusum.detect_cusum()
    # x = np.random.randn(300)
    # x[100:200] += 6
    # detect_cusum(x, 4, 1.5, True, True)
    # x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
    # ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)
