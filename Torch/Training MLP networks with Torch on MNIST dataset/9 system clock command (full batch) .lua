require 'dp'
require 'cunn'
require 'cutorch'
require 'sys'

gputime0=sys.clock ()

-- Load the mnist data set
ds = dp.Mnist()


-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw'):cuda()
trainTargets = ds:get('train', 'targets', 'b'):cuda()
validInputs = ds:get('valid', 'inputs', 'bchw'):cuda()
validTargets = ds:get('valid', 'targets', 'b'):cuda()
testInputs = ds:get('test', 'inputs', 'bchw'):cuda()
testTargets = ds:get('test', 'targets', 'b'):cuda()

batchsize = testInputs:size(1)

-- Create a two-layer network
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax()) 
module:cuda()


-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion():cuda()

require 'optim'
-- allocate a confusion matrix
cm = optim.ConfusionMatrix(10)
-- create a function to compute 
function classEval(module, inputs, targets)
  cm:zero()
 
  for idx = 1, inputs:size(1), batchsize do
    local input = torch.Tensor(batchsize, 1, 28, 28):cuda()
    local target = torch.Tensor(batchsize):cuda()
    local i = 1
    for j = idx, math.min(idx+batchsize-1, inputs:size(1)) do
      local inputt = inputs[j]
      local targett = targets[j]
      input[i] = inputt
      target[i] = targett
      i = i+1
    end
        
    for m = 1, input:size(1) do
      local output = module:forward(input[m])
      cm:add(output, target[m])
    end
        
  end
  cm:updateValids()
  return cm.totalValid
end

require 'dpnn'
function trainEpoch(module, criterion, inputs, targets)
    
  for i = 1, inputs:size(1), batchsize do
    local input = torch.Tensor(batchsize, 1, 28, 28):cuda()
    local target = torch.Tensor(batchsize):cuda()
    local j = 1
    for n = i, math.min(i+batchsize-1, inputs:size(1)) do
      local inputt = inputs[n]
      local targett = targets[n]
      input[j] = inputt
      target[j] = targett
      j = j+1
    end     
        
    module:zeroGradParameters()
    for k = 1, input:size(1) do
    -- forward
      local output = module:forward(input[k])
      local loss = criterion:forward(output, target[k])
    -- backward
      local gradOutput = criterion:backward(output, target[k])
        
      local gradInput = module:backward(input[k], gradOutput) 
    end 
    -- update
    module:updateGradParameters(0.9) -- momentum (dpnn)
    module:updateParameters(0.1) -- W = W - 0.1*dL/dW 
  end
end
  

bestAccuracy, bestEpoch = 0, 0
wait = 0
for epoch=1,30 do
   trainEpoch(module, criterion, trainInputs, trainTargets)
   local validAccuracy = classEval(module, validInputs, validTargets)
   if validAccuracy > bestAccuracy then
      bestAccuracy, bestEpoch = validAccuracy, epoch
      --torch.save("/path/to/saved/model.t7", module)
      print(string.format("New maxima : %f @ %f", bestAccuracy, bestEpoch))
      wait = 0
   else
      wait = wait + 1
      if wait > 30 then break end
   end
end
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))

gputime1 = sys.clock ()
gputime = gputime1-gputime0
print('GPU Time: '.. (gputime*1) .. 's')