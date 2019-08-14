class BasicDynamicalEnv(object):
    
    def __init__(self, phi_start, phidot):
        self.phi_start = np.array(phi_start)
        self.phidot = np.array(phidot)
        self.phis = [phi_start]
        self.phidots = [self.phidot]
        self.phi = self.phi_start
        self.num_steps = 0
        
    def step(self):
        self.phi = self.phi + self.phidot
        #print(self.phi)
        self.phis.append(self.phi)
        self.phidots.append(self.phidot)
        self.num_steps +=1
        return (self.phi, self.phidot)
    
    def start(self):
        return (self.phi, self.phidot)
