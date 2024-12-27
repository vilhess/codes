import numpy as np 
from architecture import Architecture

class RandomSearch:
    def __init__(self, api, n_simu=10, max_iter=1000):
        self.api = api
        self.n_simu = n_simu
        self.max_iter = max_iter
        
    def search(self):
        all_histories = []

        for simu in range(self.n_simu):
            best_acc_rs = 0
            best_accs_rs_history = []

            for iter in range(self.max_iter):
                arch = Architecture(api=self.api)
                step_acc = arch.get_acc()

                if step_acc > best_acc_rs:
                    best_acc_rs=step_acc

                best_accs_rs_history.append(best_acc_rs)

            all_histories.append(best_accs_rs_history)
        return np.mean(all_histories, axis=0)