#create a weight matrix:
#could look something like this
weight = predictions.new_ones(predictions.size()) * self.smoothing_coefficient / (predictions.size(-1) - 1.)
#predictions.new_ones(predictions.size()) creates a new tensor with the same shape as predictions tensor but with all 1s, then this gets multiplied by the value that should be on everything but the target coordinate , lets say there is 7 classes and the smoothing coefficient= 0.1 that mean we need to divide 0.1 into 6 classes, so the value to multiply will become 0.1/6.0 after that you will have a matrix witht he same shape as predictions with that value everywhere
#then you need to change the value in every target position to 1-smoothing coefficient, you can also do that in a matrix way
weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing_coefficient))
#takes the weight tensor and performs scatter_ function (scatter with a underscore at the end means its performed in place, so it changes the weight tensor
#it puts (1. - self.smoothing_coefficient) in the positions of targets.unsqueeze(-1) along the -1 dimension of the matrix
#then you just need to multiply that weight matrix with the predictions matrix
losses=(-weight * predictions)
#then you have individual losses for everything so you need to sum them up along the x axis and take the mean across the y axis
loss = losses.sum(dim=-1).mean()

import torch
x= torch.tensor([[-0.8137, -2.5340, -2.9288, -3.8025, -1.7447, -1.5785, -3.8793],
        [-1.4749, -2.4777, -1.7667, -0.9723, -4.1433, -2.8999, -2.6991],
        [-4.0311, -3.6466, -0.5314, -2.2921, -3.1474, -4.5298, -1.5438],
        [-1.7562, -3.9943, -3.1034, -2.2477, -1.0494, -1.2653, -3.6480]])
t = torch.tensor([1,1,1,1])
smoothing_coefficient = 0
def smoothingLabel(outputs, target ,smoothing_coefficient) :
    weight = outputs.new_ones(outputs.size()) * smoothing_coefficient / (outputs.size(-1) - 1.)
    weight.scatter_(-1, target.unsqueeze(-1), (1. - smoothing_coefficient))
    losses= -weight * outputs
    return losses.sum(dim=-1).mean()

print(smoothingLabel(x,t,0.1))
