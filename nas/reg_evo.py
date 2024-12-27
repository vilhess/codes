import random
import numpy  as np
from architecture import Architecture

class RegularizedEvolution:
    def __init__(self, api, population_size, cycles, sample_size, n_simu):
        self.api = api
        self.population_size = population_size
        self.cycles = cycles
        self.sample_size = sample_size
        self.n_simu = n_simu
        self._init_list()


    def init_population(self):
        while len(self.population)<=self.population_size:
            arch = Architecture(api=self.api)
            acc = arch.get_acc()
            self.population.append(arch)
            self.history.append(arch)
            if acc>self.best_accs[-1]: self.best_accs.append(acc)
            else: self.best_accs.append(self.best_accs[-1])

    def evolve(self):
        while len(self.history)<self.cycles:
            samples = random.sample(self.population, self.sample_size)
            best_parent = max(samples, key=lambda mod: mod.acc)
            mutated = best_parent.get_child()
            mutated_acc = mutated.get_acc()
            if mutated_acc > self.best_accs[-1]:
                self.best_accs.append(mutated_acc)
            else:
                self.best_accs.append(self.best_accs[-1])
            self.population.append(mutated)
            self.history.append(mutated)
            self.population.pop(0)

    def run(self):
        self.init_population()
        self.evolve()
        return self.best_accs[1:]
    
    def _init_list(self):
        self.population = []
        self.history = []
        self.best_accs = [0.0]
    
    def search(self):
        all_accs = []
        for simu in range(self.n_simu):
            self._init_list()
            acc_hist = self.run()
            all_accs.append(acc_hist)
        all_accs = np.mean(all_accs, axis=0)
        return all_accs