require('torch')
require('cutorch')
require('nn')
require('cunn')
require('recurrent')


-- cutorch.setDevice(2)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-nSamples', 100000, 'Number of samples')
cmd:option('-networkType', 'lstm', 'Network type')
cmd:option('-inputSize', 100, 'Neural network input size')
cmd:option('-hiddenSize', 100, 'Neural network hidden layer size')
cmd:option('-seqLength', 30, 'Sequence length')
cmd:option('-batchSize', 20, 'Batch size')
cmd:option('-cpu', false, 'Run on CPU')
cmd:text()
for k, v in pairs(cmd:parse(arg)) do _G[k] = v end

local xValues = torch.rand(nSamples, seqLength * inputSize)
local yValues = torch.rand(nSamples, hiddenSize)
if cpu ~= true then
   xValues = xValues:cuda()
   yValues = yValues:cuda()
end
local xBatches = xValues:split(batchSize, 1)
local yBatches = yValues:split(batchSize, 1)

local a = torch.Timer()
local rnn
if networkType == 'rnn' then
   rnn = nn.RNN(inputSize, hiddenSize)
elseif networkType == 'lstm' then
   rnn = nn.LSTM(inputSize, hiddenSize)
else
   error('Unkown network type!')
end
local rnn = nn.Sequential():add(rnn):add(nn.SelectTable(-1))
local criterion = nn.MSECriterion()
if cpu ~= true then
   rnn:cuda()
   criterion:cuda()
end

rnn:sequence()
rnn:training()
local input = xBatches[1]:split(inputSize, 2)
criterion:forward(rnn:forward(input), yBatches[1])
rnn:backward(input, criterion:backward(rnn.output, yBatches[1]))
if cpu ~= true then cutorch.synchronize() end
print("Setup : compile + forward/backward x 1")
print("--- " .. a:time().real .. " seconds ---")
rnn:evaluate()

a:reset()
for i = 1, #xBatches do
   if (i % 1000 == 0) then
      print(i)
   end
   rnn:forward(xBatches[i]:split(inputSize, 2))
end
if cpu ~= true then cutorch.synchronize() end
print("Forward:")
print("--- " .. nSamples .. " samples in " .. a:time().real .. " seconds (" .. nSamples / a:time().real .. " samples/s) ---")

rnn:training()
a:reset()
for i = 1, #xBatches do
   if (i % 1000 == 0) then
      print(i)
   end
   local input = xBatches[i]:split(inputSize, 2)
   criterion:forward(rnn:forward(input), yBatches[i])
   rnn:zeroGradParameters()
   rnn:backward(input, criterion:backward(rnn.output, yBatches[i]))
   rnn:updateParameters(0.01)
end
if cpu ~= true then cutorch.synchronize() end
print("Forward + Backward:")
print("--- " .. nSamples .. " samples in " .. a:time().real .. " seconds (" .. nSamples / a:time().real .. " samples/s) ---")
