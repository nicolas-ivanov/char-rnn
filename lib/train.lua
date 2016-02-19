
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
--require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
require 'lib.generate'
require 'lib.rmsprop'

local MinibatchLoader = require 'util.MinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'lib.model.LSTM'
local GRU = require 'lib.model.GRU'
local RNN = require 'lib.model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/train','data directory. Should contain the file input.txt with input data')
cmd:option('-train_file_path','data/train/dialogs_10k_ascii.txt','train file relative path')
cmd:option('-test_file_path','data/test/test_dataset_ascii.txt','test file relative path')
-- model params
cmd:option('-rnn_size', 256, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
cmd:option('-reset_frequency', '200', 'reset hidden state every n characters')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',20,'number of timesteps to unroll for')
cmd:option('-batch_size',512,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.99,'fraction of data that goes into train set')
cmd:option('-val_frac',0.01,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',5,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',2,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',2,'use OpenCL (instead of CUDA)')
cmd:text()
-- params for generate()
cmd:option('-temperature',0.5,'temperature of sampling')
cmd:option('-length',50,'number of characters to sample')

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)


-- define the model: prototypes for one timestep, then clone them in time
function define_model(vocab, vocab_size)
    local do_random_init = true
    local protos

    if string.len(opt.init_from) > 0 then
        print('loading a model from checkpoint ' .. opt.init_from)
        local checkpoint = torch.load(opt.init_from)
        protos = checkpoint.protos

        -- make sure the vocabs are the same
        local vocab_compatible = true
        local checkpoint_vocab_size = 0

        for c,i in pairs(checkpoint.vocab) do
            if not (vocab[c] == i) then
                vocab_compatible = false
            end
            checkpoint_vocab_size = checkpoint_vocab_size + 1
        end

        if not (checkpoint_vocab_size == vocab_size) then
            vocab_compatible = false
            print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
        end

        assert(vocab_compatible, 'error, the character vocabulary for this dataset and ' ..
                'the one in the saved checkpoint are not the same. This is trouble.')
        -- overwrite model settings based on checkpoint to ensure compatibility
        print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' ..
                checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
        opt.rnn_size = checkpoint.opt.rnn_size
        opt.num_layers = checkpoint.opt.num_layers
        opt.model = checkpoint.opt.model
        do_random_init = false

    else
        print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
        protos = {}
        if opt.model == 'lstm' then
            protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
        elseif opt.model == 'gru' then
            protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
        elseif opt.model == 'rnn' then
            protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
        end
        protos.criterion = nn.ClassNLLCriterion()
    end

    return protos, do_random_init
end

-- the initial state of the cell/hidden states
function init_hidden_state()
    local init_state = {}

    for L=1,opt.num_layers do
        local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
        if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(init_state, h_init:clone())
        if opt.model == 'lstm' then
            table.insert(init_state, h_init:clone())
        end
    end

    return init_state
end


-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
function init_lstm_forget_gates(protos)
    if opt.model == 'lstm' then
        for layer_idx = 1, opt.num_layers do
            for _,node in ipairs(protos.rnn.forwardnodes) do
                if node.data.annotations.name == "i2h_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
                end
            end
        end
    end
    return protos
end


-- make a bunch of clones after flattening, as that reallocates memory
function get_proto_clones(protos)
    local clones = {}
    for name, proto in pairs(protos) do
        print('cloning ' .. name)
        clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
    end
    return clones
end


-- preprocessing helper function
local function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    return x,y
end


local function reset_state(init_state)
    local rnn_state = {[0] = init_state}
    return rnn_state
end


-- evaluate the loss over an entire split
local function eval_split(split_index, data_loader, init_state, clones)
    print('evaluating loss over split index ' .. split_index)
    local n = data_loader.split_sizes[split_index]

    data_loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = reset_state(init_state)
    
    for i = 1,n do -- iterate over batches in the split
        if i % opt.reset_frequency == 0 then
            rnn_state = reset_state(init_state)
        end

        -- fetch a batch
        local x, y = data_loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            local prediction = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end
        -- carry over lstm state
        -- TODO: figure out what index 0 is doing here
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end


local function get_elapsed_time(timer)
--    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
--        --[[
--        Note on timing: The reported time can be off because the GPU is invoked async. If one
--        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
--        I will avoid doing so by default because this can incur computational overhead.
--        --]]
--        cutorch.synchronize()
--    end
    local elapsed_time = timer:time().real
    return elapsed_time
end


local function update_decay_rate(learningRate, loader, epoch, iteration_num)
    -- exponential learning rate decay
    if iteration_num % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            learningRate = learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. learningRate)
        end
    end

    return learningRate
end


local function save_checkpoint(protos, opt, train_losses, val_losses, epoch, iteration_num, data_loader, init_state, clones)
    -- evaluate loss on validation data
    local val_loss = eval_split(2, data_loader, init_state, clones) -- 2 = validation
    val_losses[iteration_num] = val_loss

    local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.val_loss = val_loss
    checkpoint.val_losses = val_losses
    checkpoint.i = iteration_num
    checkpoint.epoch = epoch
    checkpoint.vocab = data_loader.vocab_mapping
    torch.save(savefile, checkpoint)

    return checkpoint
end


local function is_sanity_check_passed(loss, loss0)
    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, ' ..
                'or create a new issue, if none exist.  Ideally, please state: ' ..
                'your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        return false
    end

    if loss0 == nil then loss0 = loss[1] end

    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        return false
    else
        return true
    end
end


-- do fwd/bwd and return loss, grad_params
function feval(x_params, packed_args)
    local params, grad_params, clones, init_state, init_state_global, data_loader = unpack(packed_args)
    if x_params ~= params then
        params:copy(x_params)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = data_loader:next_batch(1)
    x,y = prepro(x,y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
            -- note we do k-1 because first item is dembeddings, and then follow the
            -- derivatives of the state, starting at index 2. I know...
            drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end


function train(data_loader, protos, params, grad_params, clones, init_state, init_state_global)
    -- start optimization here
    local train_losses = {}
    local val_losses = {}
    local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
    local iterations = opt.max_epochs * data_loader.ntrain
    local iterations_per_epoch = data_loader.ntrain
    local loss0

    local first_epoch_percentage = 1 / data_loader.ntrain
    local first_iteration_num = 1
    local current_checkpoint = save_checkpoint(protos, opt, train_losses, val_losses, first_epoch_percentage, first_iteration_num, data_loader, init_state, clones)

    for i = 1, iterations do
        local epoch = i / data_loader.ntrain
        local timer = torch.Timer()
        local packed_args = {params, grad_params, clones, init_state, init_state_global, data_loader }
        local _, loss = rmsprop(feval, params, optim_state, nil, packed_args)
        local time = get_elapsed_time(timer)

        local train_loss = loss[1] -- the loss is inside a list, pop it
        train_losses[i] = train_loss
        optim_state.learningRate = update_decay_rate(optim_state.learningRate, data_loader, epoch, i)

        -- every now and then or on last iteration
        if i % opt.eval_val_every == 0 or i == iterations then
            current_checkpoint = save_checkpoint(protos, opt, train_losses, val_losses, epoch, i, data_loader, init_state, clones)
            generate_test_responses(opt.test_file_path, current_checkpoint)
        end

        if i % opt.print_every == 0 then
            print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs",
                i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
        end

        if i % 10 == 0 then collectgarbage() end

        if not is_sanity_check_passed(loss, loss0) then break end
    end
end

function main()
    check_cunn_availability()
    check_clnn_availability()

    -- train / val / test split for data, in fractions
    local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
    local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

    -- create the data loader class
    local data_loader = MinibatchLoader.create(opt.data_dir, opt.train_file_path, opt.batch_size, opt.seq_length, split_sizes)
    local vocab_size = data_loader.vocab_size  -- the number of distinct characters
    local vocab = data_loader.vocab_mapping
    print('vocab size: ' .. vocab_size)

    -- make sure output directory exists
    if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end


    local protos, do_random_init = define_model(vocab, vocab_size)
    local init_state = init_hidden_state()
    model_utils.transfer_model_to_gpu(protos)

    -- put the above things into one flattened parameters tensor
    local params, grad_params = model_utils.combine_all_parameters(protos.rnn)
    print('number of parameters in the model: ' .. params:nElement())

    -- initialization
    if do_random_init then
        params:uniform(-0.08, 0.08) -- small uniform numbers
    end

    protos = init_lstm_forget_gates(protos)
    local clones = get_proto_clones(protos)
    local init_state_global = clone_list(init_state)

    train(data_loader, protos, params, grad_params, clones, init_state, init_state_global)
end

main()