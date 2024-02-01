import numpy as np
import matplotlib.pyplot as plt
class Kmeans:
    def __init__(self,data,k):
        self.data=data  #定义实例属性
        self.k = k
 
    def kmeans(self):
        dataset = np.random.random(size=self.k)
        dataset = np.floor(dataset * len(self.data))
        dataset = dataset.astype(int)
        print('Random index', dataset)  #k points in the data set were randomly selected as the initial center points
        center = data[dataset]
        cls = np.zeros([len(self.data)], np.int)
        print('Init-center=\n', center)
        run = 1
        time = 0
        n = len(self.data)
        while run:
            time += 1
            for i in range(n):
                tmp = data[i] - center  #delta
                tmp = np.square(tmp)  #square
                tmp = np.sum(tmp, axis=1)  #Sum by row
                cls[i] = np.argmin(tmp)
            run = 0
            # Compute the center point of each class
            for i in range(self.k):
                club = data[cls == i] #Find all the points in this class
                new = np.mean(club, axis=0)
                instance = np.abs(center[i] - new)
                if np.sum(instance, axis=0) > 1e-4:   #If the distance of the new center is small, it can be regarded as the same class
                    center[i] = new
                    run = 1
            print('new center=\n', center)
        print('The number of iterations:', time)
        # get different kinds of graphs
        for i in range(self.k):
            club = data[cls == i]
            self.show(club)
        #final center
        self.show(center)
        print('cluster labels:')
        print(cls)
 
    def show(self,data):  #draw graph
        x = data.T[0]
        y = data.T[1]
        plt.scatter(x, y, s=300, c='pink', edgecolors='w', marker='*')
        plt.show()
 
if __name__ == '__main__':
    data = np.random.rand(20,2)
    K = 5
    model = Kmeans(data,K)
    model.kmeans()