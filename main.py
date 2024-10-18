import numpy as np
class nn:
    def __init__(self):
        self.weightsandbias()
        self.learningrate = 0.00000001
        self.input = np.array([[20],[30],[44]])
        self.output = np.array([[2],[3],[4.4]])
    def makeweightsandbias(self, inp, out):
        return np.random.rand(inp, out), np.zeros(out)
    def weightsandbias(self):
        self.h1w, self.h1b = self.makeweightsandbias(1, 20)
        self.h2w, self.h2b = self.makeweightsandbias(20, 20)
        self.ow, self.ob = self.makeweightsandbias(20, 1)
    def weightsandbiascustom(self,hiddenlayercount,hidden):
        self.hiddenlayercount=hiddenlayercount
        self.hiddenlayerweights=[[[0]] for _ in range*(hiddenlayercount)]
        self.hiddenlayerbias=[[[0]] for _ in range*(hiddenlayercount)]
        for i in range(hiddenlayercount):
            a=np.array( [self.makeweightsandbias(hidden[i],hidden[i+1])])
            self.hiddenlayerweights,self.hiddenlayerbias=a[0],a[1]
    def fpasscustom(self,input):
        self.returna=[]
        for _ in range(len(self.hiddenlayerweights)):
            try:
                self.returna.append(self.relu(self.returna[-1]))
            except:
                self.returna.append(self.relu(input))
    def opasscustom(self,x,y):
        _ =self.fpasscustom(x)
        error=[]
        derrivativ=[]
        error.append(self.returna[-1]-y)
        derrivativ.append(error[0])
        for i in range(self.hiddenlayercount):
           error.append(derrivativ[-1].dot(self.hiddenlayerweights[i].T))
           derrivativ.append(error[-1]*self.drelu(self.hiddenlayerbias[i]))
        for i in range(self.hiddenlayercount):
            self.hiddenlayerweights[len(self.hiddenlayerweights)-i]-=self.learningrate*self.hiddenlayerbias[i].T.dot(derrivativ[i])
            self.hiddenlayerbias[len(self.hiddenlayerbias)-i]-=self.learningrate*np.sum(derrivativ[i],axis=0)
        

        
    def relu(self, x):
        return np.maximum(0, x)
    def drelu(self, x):
        return np.where(x > 0, 1, 0)
    def fpass(self, x):
        self.a1 = x.dot(self.h1w) + self.h1b
        self.a2 = self.relu(self.a1)
        self.b1 = self.a2.dot(self.h2w) + self.h2b
        self.b2 = self.relu(self.b1)
        self.c1 = self.b2.dot(self.ow) + self.ob
        self.c2 = self.c1
        return self.c1
    def opass(self, x, y):
        _=self.fpass(x)
        error = self.c1-y 
        dout = error        
        h2error = dout.dot(self.ow.T)
        h2d = h2error * self.drelu(self.b2)
        h1error = h2d.dot(self.h2w.T)
        h1d = h1error * self.drelu(self.a1)
        self.ow -= self.learningrate * self.b2.T.dot(dout)
        self.h2w -= self.learningrate * self.a2.T.dot(h2d)
        self.h1w -= self.learningrate * x.T.dot(h1d)
        self.ob -= self.learningrate * np.sum(dout, axis=0)
        self.h2b -= self.learningrate * np.sum(h2d, axis=0)
        self.h1b -= self.learningrate * np.sum(h1d, axis=0)
        print("error:", error)
        print("output:", self.c2)
a=nn()
for i in range(1000000):
    a.opass(a.input,a.output)
print(a.fpass(np.array([40])))
