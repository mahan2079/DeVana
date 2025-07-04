# ... imports ...
from skopt import Optimizer

class BayesOptWorker(QThread):
    finished = pyqtSignal(dict, list, list, float)
    error    = pyqtSignal(str)
    update   = pyqtSignal(str)

    def __init__(self, main_params, target_values, weights,
                 omega_start, omega_end, omega_points,
                 bayes_iters=60, param_data=None):
        super().__init__()
        self.main_params  = main_params
        self.target_vals  = target_values
        self.weights      = weights
        self.omega_start  = omega_start
        self.omega_end    = omega_end
        self.omega_points = omega_points
        self.bayes_iters  = bayes_iters
        self.param_data   = param_data or []   # (name, low, high, fixed)
        # build skopt search-space
        self.space, self.fixed = [], {}
        for i,(n,lo,hi,fixed) in enumerate(self.param_data):
            if fixed: self.fixed[i]=lo
            self.space.append((lo,hi))
        self.opt = Optimizer(self.space, acq_func="EI")

    def frf_fitness(self, x):
        # apply fixed params
        for i,val in self.fixed.items():
            x[i] = val
        res = frf(main_system_parameters=self.main_params,
                  dva_parameters=tuple(x),
                  omega_start=self.omega_start,
                  omega_end=self.omega_end,
                  omega_points=self.omega_points,
                  # ... pass target/weights ...
                  plot_figure=False, show_peaks=False, show_slopes=False)
        return abs(res.get('singular_response', 1e6) - 1)   # minimise

    def run(self):
        best_x, best_y = None, 1e9
        try:
            for k in range(1, self.bayes_iters+1):
                x = self.opt.ask()
                y = self.frf_fitness(x)
                self.opt.tell(x, y)
                if y < best_y:
                    best_x, best_y = x[:], y
                    self.update.emit(f"Iter {k}: new best {best_y:.4g}")
            self.finished.emit({'singular_response':best_y},
                               best_x, [p[0] for p in self.param_data], best_y)
        except Exception as e:
            self.error.emit(str(e))