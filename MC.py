import data
import dice
import math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

class MC:

    def __init__(self):
        self.dim = 5 #dimension as states

        d = data.Data()
        self.d = d.get_d()

        self.p1 = np.zeros(shape=(self.dim, self.dim))
        self.f1 = np.zeros(shape=(self.dim, self.dim))
        self.p = np.zeros(shape=(self.dim*self.dim, self.dim))
        self.f = np.zeros(shape=(self.dim*self.dim, self.dim))

    # First Order Markov Chain
    def mc1(self, data_set):
        self.p1 = np.zeros(shape=(self.dim, self.dim))
        self.f1 = np.zeros(shape=(self.dim, self.dim))
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]-1):
                if data_set[i][j+1] > 0:
                    row = int(data_set[i][j]) - 1
                    col = int(data_set[i][j+1]) - 1
                    self.f1[row][col] += 1
                    self.f1[row][col] = int(self.f1[row][col])

        for i in range(self.dim):
            Sum = sum(self.f1[i,:])
            for j in range(self.dim):
                self.p1[i][j] = self.f1[i][j] / Sum

        print(self.p1)
        return self.f1, self.p1
    
    # Reduced Trnsition Matrix
    def rtm(self, data_set):
        self.p = np.zeros(shape=(self.dim*self.dim, self.dim))
        self.f = np.zeros(shape=(self.dim*self.dim, self.dim))
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]-2):
                if data_set[i][j+2] > 0:
                    last2 = int(data_set[i][j])
                    last1 = int(data_set[i][j+1])
                    last0 = int(data_set[i][j+2])
                    row = last1 - 1 + (last2-1) * self.dim
                    col = last0 - 1
                    self.f[row][col] += 1
                    self.f[row][col] = int(self.f[row][col])

        for i in range(self.dim*self.dim): #row
            Sum = sum(self.f[i,:])
            if Sum != 0:
                self.p[i,:] = self.f[i,:] / Sum

        print(self.p)
        return self.f, self.p        

    def get_data(self):
        print(self.d[-1])

    def dependency(self, mc=1):
        alpha = 0
        if mc == 1:
            for i in range(self.dim):
                for j in range(self.dim):
                    alpha += self.f1[i][j] * math.log1p(self.p1[i][j]/np.sum(self.p1, axis=0)[j])
        else:
            for i in range(self.dim*self.dim):
                for j in range(self.dim):
                    alpha += self.f[i][j] * math.log1p(self.p[i][j]/np.sum(self.p, axis=0)[j])

        print(2 * alpha)
        return 2 * alpha

    def temporary_stationary(self):
        beta = 0
        # set the interval as 22 days
        d1 = self.d[0:22]
        temp_p = [self.mc1(d1)] #[([f1], [p1]), ([f2], [p2]), ...]
        d2 = self.d[22:44]
        temp_p.append(self.mc1(d2))
        d3 = self.d[44:66]
        temp_p.append(self.mc1(d3))
        # get the orinigal p for the whole dataset
        self.mc1(self.d)
        print(self.p1)

        for k in range(3):
            for i in range(self.dim):
                for j in range(self.dim):
                    beta += temp_p[k][0][i][j] * math.log1p(temp_p[k][1][i][j]/self.p1[i][j])

        print(2*beta)
        return 2*beta

    def test(self):
        # First order markov chain
        FirstOrderMarkovChain = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.1,0.4,0.5]])
        P1 = np.zeros(9).reshape(3,3)
        Output1 = [1]
        dim = FirstOrderMarkovChain.shape[0]
        d = dice.Dice(dim)

        for i in range(5000):
            last = Output1[-1]
            prob = [0]
            for i in range(dim-1):
                prob.append(FirstOrderMarkovChain[last-1][i]+prob[-1])
            prob.append(1)
            d.set_bounds(prob)
            Output1.append(d.roll())

        for j in range(len(Output1)-1):
            row = Output1[j]-1
            col = Output1[j+1]-1
            P1[row][col]+=1
            
        for k in range(dim):
            Sum = sum(P1[k,:])
            for m in range(dim):
                P1[k][m] /=Sum

        print("First Order Input:\n", FirstOrderMarkovChain)
        print("First Order Output:\n", P1)

        # Second order markov chain
        SecondOrderMarkovChain = np.dot(FirstOrderMarkovChain,FirstOrderMarkovChain)
        P2 = np.zeros(9).reshape(3,3)

        for j in range(len(Output1)-2):
           row = Output1[j]-1
           col = Output1[j+2]-1
           P2[row][col]+=1
           
        for k in range(dim):
           Sum = sum(P2[k,:])
           for m in range(dim):
               P2[k][m] /=Sum

        print("Second Order Input:\n", SecondOrderMarkovChain)
        print("Second Order Output:\n", P2)

        # Reduced Transition Matrix
        ReducedTransitionMatrix = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.1,0.4,0.5], 
                                           [0.1,0.8,0.1],[0.2,0.4,0.4],[0.1,0.7,0.2],
                                           [0.4,0.3,0.3],[0.2,0.1,0.7],[0.5,0.1,0.4]])
        P = np.zeros(27).reshape(9,3)
        Output = [1,2]
        dim0 = ReducedTransitionMatrix.shape[0] #9
        dim1 = ReducedTransitionMatrix.shape[1] #3
        d = dice.Dice(dim1)

        for i in range(10000):
            last1 = Output[-1]
            last2 = Output[-2]
            prob = [0]
            for i in range(dim1-1):
                idx = last2-1+(last1-1)*3
                prob.append(ReducedTransitionMatrix[idx][i]+prob[-1])
            prob.append(1)
            d.set_bounds(prob)
            Output.append(d.roll())

        for j in range(len(Output)-2):
            last2 = Output[j]
            last1 = Output[j+1]
            last0 = Output[j+2]
            row = last2-1+(last1-1)*3
            col = last0-1
            P[row][col]+=1

        for k in range(dim0): #row  
            Sum = sum(P[k,:])
            P[k,:] /= Sum

        print("Reduced Transition Matrix Input:\n", ReducedTransitionMatrix)
        print("Reduced Transition Matrix Output:\n", P)
           

if __name__ == "__main__": 
    d = data.Data()
    fd = d.get_d()

    mc = MC()
    mc.mc1(fd)
    #mc.dependency()
    #mc.temporary_stationary()

    
    #mc.rtm(fd)
    #mc.dependency(0)