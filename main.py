import numpy as np
from sklearn import datasets
import random
from sklearn.cluster import KMeans

iris = datasets.load_iris()
data = iris.data


class Chromosome:
    def __init__(self, k, array):
        self.cluster_count = k
        self.length = 150
        self.genotype = np.copy(array)

    def eval(self):
        clustered = []
        centroids = []
        for i in range(self.cluster_count):
            clustered.append([])
            for j in range(self.length):
                if self.genotype[j] == i:
                    clustered[i].append(j)
        for i in clustered:
            avg = [0, 0, 0, 0]
            size = 0
            for j in i:
                for k in range(4):
                    avg[k] += data[j][k]
                size += 1
            if size == 0:
                size = 1
            for j in range(4):
                avg[j] = avg[j]/size
            centroids.append(avg)
        avg = [0, 0, 0, 0]
        counter = 0
        for i in clustered:
            for j in i:
                for k in range(4):
                    avg[k] += (data[j][k] - centroids[counter][k])
            counter += 1
        for j in range(4):
            avg[j] = avg[j] / self.length
        sum_ = 0
        for i in avg:
            sum_ += i
        return [sum_]

    def mutate(self):
        for i in range(4):
            cell = random.randint(0, self.length - 1)
            self.genotype[cell] = random.randint(0, self.cluster_count - 1)

    def crossover_partition(self, n):
        if n == 1:
            return self.genotype[:20]
        else:
            return self.genotype[-130:]


class Population:
    def __init__(self, k):
        self.eval_avg = 0
        self.pop = []
        self.length = 150
        self.k = k
        self.count = 0

    def cluster(self):
        for i in range(10):
            new_gen_seq = []
            for j in range(150):
                new_gen_seq.append(random.randint(0, self.k - 1))
            new_gen = Chromosome(self.k, new_gen_seq)
            self.pop.append(new_gen)
        while not self.terminate():
            self.reproduce()
            self.mutate()
            self.eliminate()
            self.count += 1
        print("genetic algorithm labels:\n", self.best_genome())

    def reproduce(self):
        for i in range(20):
            num_ = random.randint(0, len(self.pop) - 3)
            self.crossover(num_, num_ + 2)

    def crossover(self, n, m):
        new_genotype = self.pop[n].crossover_partition(1) + self.pop[m].crossover_partition(2)
        new_ch = Chromosome(self.k, new_genotype)
        self.pop.append(new_ch)

    def mutate(self):
        for i in range(3):
            num_ = random.randint(0, len(self.pop) - 1)
            self.pop[num_].mutate()

    def eliminate(self):
        eval_list = []
        for i in range(len(self.pop)):
            eval_list.append((self.pop[i].eval(), i))
        eval_list = sorted(eval_list, key=lambda x: x[0])
        eval_list = eval_list[:5]
        for i in eval_list:
            del self.pop[i[1]]

    def terminate(self):
        if len(self.pop) < 5:
            return True
        if self.count > 1000:
            return True
        population_data = np.concatenate([chromosome.eval() for chromosome in self.pop])
        population_variance = np.var(population_data)
        if population_variance < 0.0001:
            return True
        return False

    def best_genome(self):
        eval_list = []
        for i in range(len(self.pop)):
            eval_list.append((self.pop[i].eval(), i))
        eval_list = sorted(eval_list, key=lambda x: x[0])
        best_ = eval_list[-1:]
        return self.pop[best_[0][1]].genotype


pop1 = Population(5)
pop1.cluster()
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data)
kmeans_labels = kmeans.labels_
print("\nk-means labels:\n", kmeans_labels)
