import torch

class GreedySearch:
    def __init__(self, model, x):
        self.model = model
        self.x = x

    def greedy_search(self, x, max_steps=10):
        model_output = self.model(x)
        predictions = []
        
        for step in range(max_steps):
            action = torch.argmax(model_output, dim=-1)
            predictions.append(action.item())
        
        return predictions