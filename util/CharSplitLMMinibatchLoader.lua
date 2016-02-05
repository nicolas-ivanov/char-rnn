
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

EOS_SYBOL = '|'
PAD_SYBOL = '~'

local CharSplitLMMinibatchLoader = {}
CharSplitLMMinibatchLoader.__index = CharSplitLMMinibatchLoader

function CharSplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, CharSplitLMMinibatchLoader)

    self.batch_size = batch_size
    self.seq_length = seq_length

    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local x_tensor_file = path.join(data_dir, 'x_tensor.t7')
    local y_tensor_file = path.join(data_dir, 'y_tensor.t7')


    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end

    if run_prepro then
        local vocab_mapping = CharSplitLMMinibatchLoader.get_vocab_mapping(input_file)
        local tokenized_lines = CharSplitLMMinibatchLoader.get_processed_lines(input_file)
        local all_x_sequences, all_y_sequences = CharSplitLMMinibatchLoader.get_all_xy_sequences(tokenized_lines, seq_length)

        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')

        local x_tensor = CharSplitLMMinibatchLoader.sequences_to_tensor(all_x_sequences, seq_length, vocab_file)
        local y_tensor = CharSplitLMMinibatchLoader.sequences_to_tensor(all_y_sequences, seq_length, vocab_file)

        -- save output preprocessed files
        print('saving ' .. vocab_file)
        torch.save(vocab_file, vocab_mapping)

        print('saving ' .. x_tensor_file)
        torch.save(y_tensor_file, x_tensor)

        print('saving ' .. y_tensor_file)
        torch.save(y_tensor_file, y_tensor)
    end


    ----------------------------------------------------------------

    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do
        self.vocab_size = self.vocab_size + 1
    end

    ----------------------------------------------------------------
    print('loading data files...')
    local x_data = torch.load(x_tensor_file)
    local y_data = torch.load(y_tensor_file)

    local samples_num = x_data.size(0)
    local trimmed_samples_num = math.floor(samples_num / self.batch_size) * self.batch_size
    self.x_batches = table.slice(x_data, 1, trimmed_samples_num)
    self.y_batches = table.slice(y_data, 1, trimmed_samples_num)

    self.nbatches = #self.x_batches
    assert(#self.x_batches == #self.y_batches)


    ----------------------------------------------------------------

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end


function CharSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end


function CharSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end


function CharSplitLMMinibatchLoader.get_chars_set(in_textfile)
    print('loading text file...')
    local str_line
    local tot_len = 0
    local f = assert(io.open(in_textfile, "r"))

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    
    -- record all characters to a set
    local unordered_chars = {}
    str_line = f:read()
    repeat
        for char in str_line:gmatch'.' do
            if not unordered_chars[char] then unordered_chars[char] = true end
        end
        tot_len = tot_len + #str_line
        str_line = f:read()
    until not str_line
    f:close()

    return unordered_chars
end


function CharSplitLMMinibatchLoader.get_vocab_mapping(in_textfile)
    local unordered_chars = CharSplitLMMinibatchLoader.get_chars_set(in_textfile)

    -- sort into a table (i.e. keys become 1..N)
    local ordered_chars = {}
    for char in pairs(unordered_chars) do ordered_chars[#ordered_chars + 1] = char end
    table.sort(ordered_chars)

    -- invert `ordered_chars` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered_chars) do
        vocab_mapping[char] = i
    end

    return vocab_mapping
end


function CharSplitLMMinibatchLoader.get_processed_lines(input_file)
    local processed_lines = {}
    local f = assert(io.open(input_file, "r"))

    for line_str in f do
        processed_lines[#processed_lines + 1] = line_str .. EOS_SYBOL
        line_str = f:read()
    end

    f:close()
    return processed_lines
end


function CharSplitLMMinibatchLoader.pad_from_left(sentence, context_len)
    if #sentence >= context_len then
        return sentence
    else
        local padded_str = string.rep(PAD_SYBOL, context_len - #sentence) .. sentence
        return padded_str
    end
end


function CharSplitLMMinibatchLoader.get_rolling_windows(sentence, window_size)
    local rolling_windows = {}
    local cur_window = string.sub(sentence, 1, window_size)
    rolling_windows[#rolling_windows + 1] = cur_window

    for new_char in string.sub(sentence, window_size, #sentence) do
        cur_window = string.sub(cur_window, 2, #sentence) .. new_char
        rolling_windows[#rolling_windows + 1] = cur_window
    end

    return rolling_windows
end


function CharSplitLMMinibatchLoader.get_all_xy_sequences(processed_lines, seq_length)
    local x_sequences = {}
    local y_sequences = {}

    for i=1, #processed_lines, 2 do
        local padded_sentence = CharSplitLMMinibatchLoader.pad_from_left(processed_lines[i], seq_length)
        local next_sentence = processed_lines[i+1]
        local joined_sent = padded_sentence .. next_sentence

        local rolling_windows = CharSplitLMMinibatchLoader.get_rolling_windows(joined_sent, seq_length + 1)

        for window in rolling_windows do
            local x_seq = string.sub(window, 1, #window-1)
            local y_seq = string.sub(window, 2, #window)
            x_sequences[#x_sequences + 1] = x_seq
            y_sequences[#y_sequences + 1] = y_seq
        end
    end
    return x_sequences, y_sequences
end


local function sequences_to_tensor(input_sequences, seq_length, vocab_mapping)
    print('putting data into tensor...')
    local data_tensor = torch.Tensor(#input_sequences, seq_length)

    for seq_id=1, #input_sequences do
        for char_id=1, seq_length do
            local current_char = input_sequences[seq_id]:sub(char_id, char_id) -- lua has no string indexing using []
            data_tensor[seq_id][char_id] = vocab_mapping[current_char]
        end
    end

    return data_tensor
end

return CharSplitLMMinibatchLoader