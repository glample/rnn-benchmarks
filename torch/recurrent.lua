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
local xBatches = xValues:view(-1, batchSize, seqLength, inputSize)
local yBatches = yValues:view(-1, batchSize, hiddenSize)

local rnn
if networkType == 'rnn' then
   rnn = nn.RNN(inputSize, hiddenSize)
elseif networkType == 'lstm' then
   rnn = nn.LSTM(inputSize, hiddenSize)
else
   error('Unkown network type!')
end

rnn:sequence()
rnn:setIterations(seqLength)

local rnn = nn.Sequential():add(rnn):add(nn.Select(2, seqLength))
local criterion = nn.MSECriterion()
if cpu ~= true then
   rnn:cuda()
   criterion:cuda()
end

local start = os.clock()
for i = 1, xBatches:size(1) do
   if (i % 100 == 0) then
      print(i)
   end
   rnn:forward(xBatches[i])
end
print("Forward:")
print("--- " .. nSamples .. " samples in " .. (os.clock() - start) .. " seconds (" .. nSamples / (os.clock() - start) .. " samples/s) ---")

rnn:sequence()
rnn:training()

start = os.clock()
for i = 1, xBatches:size(1) do
   if (i % 100 == 0) then
      print(i)
   end
   local input = xBatches[i]
   criterion:forward(rnn:forward(input), yBatches[i])
   rnn:zeroGradParameters()
   rnn:backward(input, criterion:backward(rnn.output, yBatches[i]))
   rnn:updateParameters(0.01)
end
print("Forward + Backward:")
print("--- " .. nSamples .. " samples in " .. (os.clock() - start) .. " seconds (" .. nSamples / (os.clock() - start) .. " samples/s) ---")
