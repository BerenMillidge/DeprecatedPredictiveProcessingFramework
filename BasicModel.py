
class Model():
    
    def __init__(self, data, epochs, convergence_runs=200):
        self.convergence_runs = 200
        self.data = data
        self.epochs = epochs
        self.layers = []
        self.upwards = []
        self.downwards = []
        self.wlist = []
        self.predlist = []
        self.pelist = []
        self.full_predlist = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def initialize(self):
        for i, layer in enumerate(self.layers):
            self.upwards.append(np.random.normal(0,0.1, [layer.top_down_size,1]))
            self.downwards.append(np.random.normal(0,0.1, [layer.bottom_up_size,1]))
            self.wlist.append([])
            self.predlist.append([])
            self.pelist.append([])
        self.upwards.append(0)
        self.downwards.append(None)
            
    def train(self):
        for epoch in range(self.epochs):
            print("Starting epoch: " + str(epoch))
            self.downwards[-1] = None
            for dat in self.data:
                self.upwards[0] = np.reshape(dat,(len(dat),1)) 
                for i in range(self.convergence_runs):
                    for j,layer in enumerate(self.layers):
                        #print("Layer " + str(j))
                        up, down, weights, preds, pes = layer.run(self.upwards[j], self.downwards[j+1])
                        #print(up)
                        #print(down)
                        self.upwards[j+1] = up
                        self.downwards[j] = down # should/does this work?
                        self.wlist[j].append(np.sum(weights))
                        self.predlist[j].append(np.sum(preds))
                        self.pelist[j].append(np.sum(pes))
                        
            self.full_predlist.append(np.copy(self.layers[0].preds))
            
            
    def get_latents(self, inputs):
        latents = []
        for inp in inputs:
            for i in range(self.convergence_runs):
                self.upwards[0] = np.reshape(inp, (len(inp),1))
                for j,layer in enumerate(self.layers):
                    #print("Layer : " + str(j))
                    up, down, weights, preds, pes = layer.run(self.upwards[j], self.downwards[j+1], training=False) # as only inference!
                    self.upwards[j+1] = up
                    self.downwards[j] = down 
            latents.append(np.copy(self.layers[-1].latents)) 
        return latents
    
    def get_predictions_from_latents(self, latents):
        predictions = []
        for l in latents:
            l1 = self.layers[0]
            pred = l1.top_down_prediction(l1.weights, l)
            predictions.append(pred)
            
        return predictions
    
    def get_predictions_from_data(self,data, plot=True):
        latents = self.get_latents(data)
        preds = self.get_predictions_from_latents(latents)
        if plot:
            for dat, pred in zip(data, preds):
                plt.imshow(np.reshape(dat, (32,32)))
                plt.show()
                plt.imshow(np.reshape(dat, (32,32)))
                plt.show()
        return preds
    
    def interpolation_latents(self,l1, l2, num_steps):
        interps = []
        diff = (l2 - l1) / num_steps
        for i in range(num_steps):
            latent = l1 + (i * diff)
            for i in range(self.convergence_runs):
                self.upwards[0] = np.reshape(latent, (len(latent),1))
                for j,layer in enumerate(self.layers):
                    #print("Layer : " + str(j))
                    up, down, weights, preds, pes = layer.run(self.upwards[j], self.downwards[j+1], training=False) # as only inference!
                    self.upwards[j+1] = up
                    self.downwards[j] = down 
            interps.append(np.copy(self.layers[-1].latents))
        return interps

    def interpolate(self, l1, l2,num_steps, plot=True):
        interps = self.interpolation_latents(l1, l2, num_steps)
        preds = self.get_predictions_from_latents(interps)
        if plot:
            sh = int(np.sqrt(len(preds[0])))
            for pred in preds:
                plt.imshow(np.reshape(pred, (sh,sh)), cmap='gray')
                plt.show()
                
        return preds
                        
    def plot(self):
        self.wlist = np.array(self.wlist[0])
        self.predlist = np.array(self.predlist[0])
        self.pelist = np.array(self.pelist[0])
        plt.plot(self.wlist)
        plt.show()
        plt.plot(self.predlist)
        plt.show()
        plt.plot(self.pelist)
        plt.show()
        for pred in self.full_predlist:
            plt.imshow(np.reshape(pred, (int(np.sqrt(len(pred))), int(np.sqrt(len(pred))))))
            plt.show()  

class Model(object):
    
    def __init__(self,env, epochs, inference_runs):
        self.env = env
        self.layers = []
        self.epochs = epochs
        self.inference_runs = inference_runs
        self.phis = []
        self.predictions = []
        self.prediction_errors = []
        self.preds = []
        self.pes = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def initialize(self):
        self.predictions.append(None)
        for layer in self.layers:
            self.predictions.append(None)
            self.prediction_errors.append(np.zeros([layer.layer_dimension, 1]))
            self.preds.append([])
            self.pes.append([])
        self.prediction_errors.append(None)
            
        
    def train(self):
        
        for i in range(self.epochs * self.inference_runs):
            if i % self.inference_runs == 0:
                self.phis = self.env.step()
                #print(self.phis)
            
            for j, layer in enumerate(self.layers):
                if i %self.inference_runs == 0:
                    pred, up, down = layer.run(self.phis[j],self.prediction_errors[j], self.predictions[j+1], learning=True)
                else:
                    pred, up, down = layer.run(self.phis[j],self.prediction_errors[j], self.predictions[j+1])
                self.prediction_errors[j+1] = up
                self.predictions[j] = down
                self.preds[j].append(np.sum(pred))
                self.pes[j].append(np.sum(up))
        return self.preds, self.pes
