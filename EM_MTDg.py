import math     
import data
import dice
import copy
import random
import numpy as np
import matplotlib.pyplot as plt 
import MTD as mtd_b

class EM_MTDg:
    def __init__(self):
        d = data.Data()
        b = mtd_b.MTD()
        self.data = d.get_d()        
        self.order = 2
        self.epsilon = 0.05
        self.dim = int(1/d.states)
        self.occurence = self.compute_occurence()[0]
        self.total_n = self.compute_occurence()[1]
        self.init_phi = np.array([b.lambda1,b.lambda2]) 
        self.init_Q = [b.q1,b.q2]
        self.final_phi = self.iteration(100)[0]
        self.final_Q = self.iteration(100)[1]        
        self.P = self.P(6)
    
    def Estimation_step(self,phi_k,Q_k):#change the number of nested loops by hand
        P = np.zeros(shape=(self.dim,self.dim,self.dim,self.order)) #Xt-2 Xt-1 Xt order
        for g in range(self.order):#step
            for s2 in range(self.dim):
                for s1 in range(self.dim):
                    for s0 in range(self.dim):
                        if g ==0:
                            ig = s1
                        else:
                            ig = s2
                        numerator = phi_k[g]*Q_k[g][ig][s0]
                        denominator = phi_k[0]*Q_k[0][s1][s0]+phi_k[1]*Q_k[1][s2][s0]
                        if denominator != 0:
                            P[s2][s1][s0][g] = numerator/denominator    
        return P

    def Maximization_step(self,P):
        phi_k = np.zeros(self.order)
        Q_k = [np.zeros(shape=(self.dim,self.dim)),np.zeros(shape=(self.dim,self.dim))]
        
        for i in range(self.order): #step
            phig = 0
            for s2 in range(self.dim):
                for s1 in range(self.dim):
                    for s0 in range(self.dim):
                        phig += P[s2][s1][s0][i]*self.occurence[s2][s1][s0]
            phig /= (self.total_n-self.order)
            phi_k[i] = phig
        
        # g = 1 
        for i in range(self.dim):
            for j in range(self.dim):# which is also s0 in the numerator 
                numerator = 0
                denominator = 0
                for s2 in range(self.dim):
                    numerator += P[s2][i][j][0]*self.occurence[s2][i][j]
                    for s0 in range(self.dim):
                        denominator += P[s2][i][s0][0]*self.occurence[s2][i][s0]
                if denominator != 0:
                    Q_k[0][i][j] = numerator/denominator  
                
        # g = 2
        for i in range(self.dim):
            for j in range(self.dim):# which is also s0 in the numerator 
                numerator = 0
                denominator = 0
                for s1 in range(self.dim):
                    numerator += P[i][s1][j][0]*self.occurence[i][s1][j]
                    for s0 in range(self.dim):
                        denominator += P[i][s1][s0][0]*self.occurence[i][s1][s0]
                if denominator != 0:
                    Q_k[1][i][j] = numerator/denominator  
        return phi_k, Q_k
    
    def iteration(self,iter_num):
        P_k = self.Estimation_step(self.init_phi,self.init_Q)
        phi_k,Q_k = self.Maximization_step(P_k)
        for i in range(iter_num):   
            old = (phi_k,Q_k)
            P_k = self.Estimation_step(phi_k,Q_k)
            phi_k,Q_k = self.Maximization_step(P_k)
            new = (phi_k,Q_k)
            delta = self.compute_loglikelihood(new[0],new[1])-self.compute_loglikelihood(old[0],old[1])
           
            if delta<self.epsilon:
                break
            
        return phi_k,Q_k
        
    def compute_loglikelihood(self,phi,Q):
        log_likelihood = 0
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]-2):
                if self.data[i][j+2]!= 0:
                    s2 = int(self.data[i][j] )-1
                    s1 = int(self.data[i][j+1]) -1
                    s0 = int(self.data[i][j+2]) -1
                    log_likelihood += math.log(phi[0]*Q[0][s1][s0]+phi[1]*Q[1][s2][s0])
        return log_likelihood

    def compute_occurence(self): #assume it's 2nd order  Xt-2 Xt-1 Xt
        occurence = np.zeros(shape=(self.dim, self.dim,self.dim))
        total_n = 0

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]-self.order):
                if self.data[i][j+2] > 0:
                    last2 = int(self.data[i][j])-1
                    last1 = int(self.data[i][j+1])-1
                    last0 = int(self.data[i][j+2])-1
                    total_n += 1
                    occurence[last2][last1][last0]+=1
                    
        return occurence,total_n    

    def BIC(self):
        # LL is the log-likelihood of the model
        # p is the number of independent parameters 
        # N the number of data
        n = self.compute_occurence()[1]
        phi_k,Q_k = self.iteration(100)
        
        ll= self.compute_loglikelihood(phi_k,Q_k)

        p_mtdg = 2*self.dim *(self.dim-1)+1
        BIC_mtdg = -2*ll+p_mtdg*math.log(n)
        
        return ll,BIC_mtdg
    
    #given sequence, predict num points afterwards
    def prediction(self,sequence,num):
        d = dice.Dice(self.dim)
        #for mtdg prediction
        sequence_mtdg = sequence[:]
        for j in range(num):
            prob = [0]
            last = sequence_mtdg[-1] 
            last2 = sequence_mtdg[-2]
            for i in range(self.dim-1):         
                p = self.final_phi[0] * self.final_Q[0][last-1][i]+self.final_phi[1] * self.final_Q[1][last2-1][i] 
                prob.append(p+prob[-1])
            
            prob.append(1)
            d.set_bounds(prob)
            sequence_mtdg.append(d.roll())

        plt.figure(figsize=(8,5),dpi = 80)
        plt.title('Prediction by MTDg model')
        plt.subplot(1,1,1)   
        plt.yticks([0,1,2,3,4,5])
        X1 = []
        Y1 = []
        idx = 0
        for i in sequence_mtdg:
            X1.append(idx)
            Y1.append(i)
            idx+=1
        
        plt.plot(X1,Y1,'red')         
        return sequence_mtdg
    
        #randomly choose 3+num points in whole data and return the whole sequence and pre-sequence
    def prediction_sequence(self,num):
        n = 3
        #i is day number and j is the time in a day 
        i = random.randint(0,len(self.data)-1)
        #need n points in the sequence and predict m afterwards
        while True:
            j=random.randint(0,len(self.data[0])-num-n)          
            if self.data[i][j+num+n-1]==0:
                continue
            else:
                sequence = self.data[i][j:j+num+n]
                break
        #only 2 points in the pre-sequence
        pre_sequence = sequence[1:3]

        return list(pre_sequence), list(sequence),(i,j)    
    
    def prediction_test(self,pre_sequence,num):
        last2 = int(pre_sequence[-2])
        last = int(pre_sequence[-1])
        
        #probability that a point is lower
        global P 
        P = [0]
        if last>1:
            for j in range(self.dim):
                p = [self.final_phi[1]*self.final_Q[0][last2-1][j]+self.final_phi[0]*self.final_Q[1][last-1][j]]
                self.pre_recursion(pre_sequence+[j+1],0,num,P,p)
   
        return P[0]
 
    def pre_recursion(self,seq,cur_step,max_step,P,p):
        if cur_step<max_step and seq[-1]>=seq[1]:
            cur_step+=1
            for i in range(self.dim):
                new_seq = copy.deepcopy(seq)
                new_seq.append(i+1)
                last2 = int(new_seq[-3]-1)
                last1 = int(new_seq[-2]-1)
                last0 = int(new_seq[-1]-1)
                new_p = copy.deepcopy(p)
                new_p[0] = new_p[0]*(self.final_phi[1]*self.final_Q[1][last2][last0]+self.final_phi[0]*self.final_Q[0][last1][last0])
                self.pre_recursion(new_seq,cur_step,max_step,P,new_p)
        else:
            if seq[-1]<seq[1]:
                P[0]= p[0]+P[0]
                
    def P(self,num):
        P_total = np.zeros(shape=(num, self.dim, self.dim))
        for k in range(num):
            for i in range(self.dim):
                for j in range(self.dim):
                    #last2 is j last is i
                    P_total[k][j][i] = self.prediction_test([j+1,i+1],k)

        return P_total
        
    def wake_up(self,num,day,time,plot = ""):
        p_wake = 0.1
        pre_sequence,sequence,(day,time)= self.prediction_sequence(num)
        last = pre_sequence[-1]
        last2 = pre_sequence[-2]
        ori_num = num
        Sleep = True
        i = 0
        while Sleep:
            last2 = int(sequence[i+1])
            last =  int(sequence[i+2])
            prob = float(self.P[num-1][last2-1][last-1])
            if prob < p_wake:
                Sleep = False  
                break 
            else:
                i += 1
                num -=1
            if num ==0:
                Sleep = False 
                break 
                
        seq = sequence[:i+3]
        wake_up_t = time+len(seq)-1
        
        if plot =="figure1":
            plt.figure(figsize=(8,5),dpi = 80)
            plt.title('Wake up time by MTDg model ')
            plt.subplot(1,1,1)   
            plt.yticks([0,1,2,3,4,5])
            
            X1 = []
            Y1 = []            
            X2 = []
            Y2 = []  
            for i in range(len(self.data[day])):
                if self.data[day][i] !=0:
                    X1.append(i)
                    Y1.append(self.d.transformed_interval[day][i])
                    if i>time and i<time+1+ori_num+3:
                        X2.append(i)
                        Y2.append(self.d.transformed_interval[day][i])
            
            plt.plot(X2,Y2,'magenta')    
            plt.plot(X1,Y1,'cadetblue')
            wake_up_t = time+len(seq)-1
            y = self.d.transformed_interval[day][wake_up_t]
            plt.plot([wake_up_t],[y],'+')
        
        elif plot =="figure2":   
            plt.figure(figsize=(8,5),dpi = 80)
            plt.title('Wake up time by MTD model ')
            plt.subplot(1,1,1)   
            plt.yticks([0,1,2,3,4,5])
            
            X1 = []
            Y1 = []            
            X2 = []
            Y2 = []     
            for i in range(len(self.data[day])):
                if self.data[day][i] !=0:
                    X1.append(i)
                    Y1.append(self.data[day][i])
                    if i>time and i<time+1+9:
                        X2.append(i)
                        Y2.append(self.data[day][i])
            
            plt.plot(X2,Y2,'magenta')    
            plt.plot(X1,Y1,'cadetblue')
            
            y = self.data[day][wake_up_t]
            plt.plot([wake_up_t],[y],'+')
                    
        return sequence,seq 
    
    def error(self,sequence,seq,num,day,time):
        #day,time,sequence,seq = self.wake_up(num,day,time)
        #print(sequence,seq)
        lowest = min(sequence[2:])
        error = seq[-1]-lowest
        return error
       
if __name__ == "__main__": 
    mtdg = EM_MTDg()
    P = mtdg.Estimation_step(mtdg.init_phi,mtdg.init_Q)
    phi_1,Q_1 = mtdg.Maximization_step(P)

#    print('loglikelihood:',em.compute_loglikelihood(em.init_phi,em.init_Q))
#    print('\n')
#    print('Iteration:')
#    ll,BIC = em.BIC()
#    print("MTDg log-likelihood:\n",ll)
#    print("MTDg BIC:\n",BIC)
#    print(em.prediction([2,2,3],10))
    
    cur_step = 0
    max_step = 6
    num = 6
    #print(mtd.prediction_test(list(m),2))
    (day,time) = mtdg.prediction_sequence(num)[2]
    #sequence,seq = mtdg.wake_up(num,day,time)
    #print("wake up:", mtd.wake_up(num,day,time,"figure1"))
    #mtdg.error(sequence,seq,num,day,time)
    
   #calculate average error of x times 
    total = 0 
    x = 100000
    for i in range(x):
        (day,time) = mtdg.prediction_sequence(num)[2]
        sequence,seq =  mtdg.wake_up(num,day,time)
        #print("wake up:", mtd.wake_up(num,day,time))
        total += mtdg.error(sequence,seq,num,day,time)
        
    total /= (x*4)
    print("MTDg:",total)
    