from PyQt5.QtCore import QThread, pyqtSignal

from modules.sobol_sensitivity import (
    perform_sobol_analysis
)



class SobolWorker(QThread):
    finished = pyqtSignal(dict, list)
    error = pyqtSignal(str)
    
    def __init__(self, main_params, dva_bounds, dva_order,
                 omega_start, omega_end, omega_points, num_samples_list,
                 target_values_dict, weights_dict, n_jobs):
        super().__init__()
        self.main_params = main_params
        self.dva_bounds = dva_bounds
        self.dva_order = dva_order
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.num_samples_list = num_samples_list
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.n_jobs = n_jobs

    def run(self):
        try:
            all_results, warnings = perform_sobol_analysis(
                main_system_parameters=self.main_params,
                dva_parameters_bounds=self.dva_bounds,
                dva_parameter_order=self.dva_order,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                num_samples_list=self.num_samples_list,
                target_values_dict=self.target_values_dict,
                weights_dict=self.weights_dict,
                visualize=False,  
                n_jobs=self.n_jobs
            )
            self.finished.emit(all_results, warnings)
        except Exception as e:
            self.error.emit(str(e))