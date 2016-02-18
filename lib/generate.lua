--[[
This file samples characters from a trained model
Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
require 'util.MinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
--cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',0.5,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end


-- check that cunn/cutorch are installed if user wants to use the GPU
function check_cunn_availability()
    local cunn, cutorch
    local ok, ok2

    if opt.gpuid >= 0 and opt.opencl == 0 then
        local ok, cunn = pcall(require, 'cunn')
        local ok2, cutorch = pcall(require, 'cutorch')
        if not ok then gprint('package cunn not found!') end
        if not ok2 then gprint('package cutorch not found!') end
        if ok and ok2 then
            gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
            gprint('Make sure that your saved checkpoint was also trained with GPU. ' ..
                'If it was trained with CPU use -gpuid -1 for sampling as well')
            cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
            cutorch.manualSeed(opt.seed)
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            opt.gpuid = -1 -- overwrite user setting
        end
    end

    return cunn, cutorch
end


-- check that clnn/cltorch are installed if user wants to use OpenCL
function check_clnn_availability()
    local clnn, cltorch
    local ok, ok2

    if opt.gpuid >= 0 and opt.opencl == 1 then
        -- TODO: report about a bug here: previously was cunn and cutorch
        local ok, clnn = pcall(require, 'clnn')
        local ok2, cltorch = pcall(require, 'cltorch')
        if not ok then print('package clnn not found!') end
        if not ok2 then print('package cltorch not found!') end
        if ok and ok2 then
            gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
            gprint('Make sure that your saved checkpoint was also trained with GPU. ' ..
                    'If it was trained with CPU use -gpuid -1 for sampling as well')
            cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
            torch.manualSeed(opt.seed)
        else
            gprint('Falling back on CPU mode')
            opt.gpuid = -1 -- overwrite user setting
        end
    end

    return clnn, cltorch
end


-- load the model checkpoint
function load_checkpoint()
    if not lfs.attributes(opt.model, 'mode') then
        gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
    end

    local checkpoint = torch.load(opt.model)
    return checkpoint
end


function get_protos()
    local protos = checkpoint.protos
    protos.rnn:evaluate() -- put in eval mode so that dropout works properly

    return protos
end


-- initialize the vocabulary (and its inverted version)
local function get_vocabs(checkpoint)
    local vocab = checkpoint.vocab
    local ivocab = {}
    for c,i in pairs(vocab) do ivocab[i] = c end

    return vocab, ivocab
end


-- initialize the rnn state to all zeros
local function init_rnn_state(checkpoint)
    gprint('creating an ' .. checkpoint.opt.model .. '...')
    local current_state = {}

    for L = 1, checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
        if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(current_state, h_init:clone())

        if checkpoint.opt.model == 'lstm' then
            table.insert(current_state, h_init:clone())
        end
    end

    return current_state
end


local function get_current_state(lst, state_size)
    local current_state = {}

    for i=1,state_size do
        table.insert(current_state, lst[i])
    end

    return current_state
end


local function get_converted_char_id(char_id)
    if opt.gpuid >= 0 and opt.opencl == 0 then char_id = char_id:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then char_id = char_id:cl() end
    return char_id
end


local function generate_response(input_str, protos, current_state, vocab, ivocab)
    local response_str = ''
    local state_size = #current_state

    -- do a few seeded timesteps to accumulate the hidden state
    local predicted_distribution
    local prev_char_id
    input_str = input_str .. EOS_SYMBOL

    for c in input_str:gmatch'.' do
        prev_char_id = torch.Tensor{vocab[c]}
        prev_char_id = get_converted_char_id(prev_char_id)
        local lst = protos.rnn:forward{prev_char_id, unpack(current_state)}

        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = get_current_state(lst, state_size)
        predicted_distribution = lst[#lst] -- last element holds the log probabilities
    end

    -- start sampling/argmaxing
    local char_num = 0
    repeat
        -- log probabilities from the previous timestep
        predicted_distribution:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(predicted_distribution):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_char_id = torch.multinomial(probs:float(), 1):resize(1):float()
        char_num = char_num + 1
        response_str = response_str .. ivocab[prev_char_id[1]]

        -- forward the rnn for next character
        local lst = protos.rnn:forward{prev_char_id, unpack(current_state)}
        current_state = get_current_state(lst, state_size)
        predicted_distribution = lst[#lst] -- last element holds the log probabilities
    until (prev_char_id[1] == vocab[EOS_SYMBOL]) or (char_num > opt.length)

    return response_str
end


function generate_test_responses(test_set_file, checkpoint)
    local vocab, ivocab = get_vocabs(checkpoint)
    local current_state = init_rnn_state(checkpoint)
    local test_set_fh = assert(io.open(test_set_file, 'r'))

    while true do
        local input_str = test_set_fh:read()
        if not input_str then break end
        local response = generate_response(input_str, checkpoint.protos, current_state, vocab, ivocab)
        print(input_str .. '\t->\t' .. response)
    end
    test_set_fh:close()
end
