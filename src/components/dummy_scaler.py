class DummyScaler:
    def scale(self, loss): 
        return loss
    def step(self, optimizer): 
        optimizer.step()
    def update(self): 
        pass