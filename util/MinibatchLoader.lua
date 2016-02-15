
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local EOS_SYBOL = '|'
local PAD_SYBOL = '~'

local LINES_BATCH_SIZE = 1000

local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

function MinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, MinibatchLoader)

    self.batch_size = batch_size
    self.seq_length = seq_length

    local input_file_path = path.join(data_dir, 'input.txt')
    local vocab_file_path = path.join(data_dir, 'vocab.t7')
    local x_tensor_file_path = path.join(data_dir, 'x_tensor.t7')
    local y_tensor_file_path = path.join(data_dir, 'y_tensor.t7')

    local run_prepro = MinibatchLoader.is_processing_needed(input_file_path, vocab_file_path, x_tensor_file_path, y_tensor_file_path)

    local x_tensor = torch.ByteTensor(1, seq_length):fill(1)
    local y_tensor = torch.ByteTensor(1, seq_length):fill(1)

    if run_prepro then
        local vocab_mapping = MinibatchLoader.get_vocab_mapping(input_file_path)

        local file = io.open(input_file_path, 'r');
        local all_lines = {}
        local all_lines_num = 0
        for line in file:lines() do
            table.insert(all_lines, line)
            all_lines_num = all_lines_num + 1
        end

        for i=1, #all_lines, LINES_BATCH_SIZE do
            local lines_chunk = {}
            print('-----------------------------------')
            print(string.format('tensor build progress: %.2f %%', 100 * (i / all_lines_num)))

            for j=1, LINES_BATCH_SIZE do
                lines_chunk[j] = all_lines[i+j]
            end

            local tokenized_lines = MinibatchLoader.get_processed_lines(lines_chunk)
            local all_x_sequences, all_y_sequences = MinibatchLoader.get_all_xy_sequences(tokenized_lines, seq_length)
--            for i=1, #all_x_sequences do
--                print(all_x_sequences[i], all_y_sequences[i])
--            end

            -- construct a tensor with all the data, and vocab file
            print('one-time setup: preprocessing input text file ' .. input_file_path .. '...')

            local x_tensor_part = MinibatchLoader.sequences_to_tensor(all_x_sequences, seq_length, vocab_mapping)
            local y_tensor_part = MinibatchLoader.sequences_to_tensor(all_y_sequences, seq_length, vocab_mapping)

            x_tensor = torch.cat(x_tensor, x_tensor_part, 1)
            y_tensor = torch.cat(y_tensor, y_tensor_part, 1)
            print('dimensions of x-tensor:', x_tensor:size(1), x_tensor:size(2))
            print('dimensions of y-tensor:', y_tensor:size(1), y_tensor:size(2))
        end

        MinibatchLoader.save_files(vocab_file_path, vocab_mapping, x_tensor_file_path, x_tensor, y_tensor_file_path, y_tensor)
    end

    print('loading data files...')

    self.vocab_mapping = torch.load(vocab_file_path)
    self.vocab_size = MinibatchLoader.get_table_len(self.vocab_mapping)

    local x_data = torch.load(x_tensor_file_path)
    local y_data = torch.load(y_tensor_file_path)

    local items_num_in_batch = self.batch_size * seq_length
    x_data = MinibatchLoader.trim_tensor(x_data, items_num_in_batch)
    y_data = MinibatchLoader.trim_tensor(y_data, items_num_in_batch)

--    print(x_data)

    self.x_batches = x_data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.y_batches = y_data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches

    self.nbatches = #self.x_batches
    assert(#self.x_batches == #self.y_batches)

--    for i = 1, #self.x_batches do
--        print(self.x_batches[i])
--    end

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


function MinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end


function MinibatchLoader:next_batch(split_index)
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


function MinibatchLoader.is_processing_needed(input_file, vocab_file, x_tensor_file, y_tensor_file)
    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) and path.exists(x_tensor_file) and path.exists(y_tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab and tensor files do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local x_tensor_attr = lfs.attributes(x_tensor_file)
        local y_tensor_attr = lfs.attributes(y_tensor_file)
        if input_attr.modification > vocab_attr.modification
                or input_attr.modification > x_tensor_attr.modification
                or input_attr.modification > y_tensor_attr.modification
        then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    return run_prepro
end


function MinibatchLoader.save_files(vocab_file, vocab_mapping, x_tensor_file, x_tensor, y_tensor_file, y_tensor)
    -- save output preprocessed files
    print('saving ' .. vocab_file)
    torch.save(vocab_file, vocab_mapping)

    print('saving ' .. x_tensor_file)
    torch.save(x_tensor_file, x_tensor)

    print('saving ' .. y_tensor_file)
    torch.save(y_tensor_file, y_tensor)
end


function MinibatchLoader.get_chars_set(in_textfile)
    print('loading text file...')
    local str_line
    local f = assert(io.open(in_textfile, "r"))

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    
    -- record all characters to a set
    local unordered_chars = {}
    unordered_chars[PAD_SYBOL] = true
    unordered_chars[EOS_SYBOL] = true

    str_line = f:read()
    repeat
        for char in str_line:gmatch'.' do
            if not unordered_chars[char] then unordered_chars[char] = true end
        end
        str_line = f:read()
    until not str_line
    f:close()

    return unordered_chars
end


function MinibatchLoader.get_vocab_mapping(in_textfile)
    local unordered_chars = MinibatchLoader.get_chars_set(in_textfile)

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


function MinibatchLoader.get_processed_lines(raw_lines)
    local processed_lines = {}

    for i=1, #raw_lines do
        local line_str = raw_lines[i]
        processed_lines[#processed_lines + 1] = line_str .. EOS_SYBOL
    end

    return processed_lines
end


function MinibatchLoader.pad_from_left(sentence, context_len)
    if #sentence >= context_len then
        return sentence
    else
        local padded_str = string.rep(PAD_SYBOL, context_len - #sentence) .. sentence
        return padded_str
    end
end


function MinibatchLoader.get_rolling_windows(sentence, window_size)
    local rolling_windows = {}
    local cur_window = sentence:sub(1, window_size)
    rolling_windows[#rolling_windows + 1] = cur_window

    local sentence_tail = sentence:sub(window_size + 1, #sentence)

    for i=1, #sentence_tail do
        local new_char = sentence_tail:sub(i, i)
        cur_window = cur_window:sub(2, #cur_window) .. new_char
        rolling_windows[#rolling_windows + 1] = cur_window
    end

    return rolling_windows
end


function MinibatchLoader.get_all_xy_sequences(processed_lines, seq_length)
    local x_sequences = {}
    local y_sequences = {}

    -- even the number of lines
    if #processed_lines % 2 == 1 then
        processed_lines[#processed_lines + 1] = EOS_SYBOL
    end

    for i=1, #processed_lines, 2 do
        local padded_sentence = MinibatchLoader.pad_from_left(processed_lines[i], seq_length)
        local next_sentence = processed_lines[i+1]
        local joined_sent = padded_sentence .. next_sentence

        local rolling_windows = MinibatchLoader.get_rolling_windows(joined_sent, seq_length + 1)

        for _, window in pairs(rolling_windows) do
            local x_seq = window:sub(1, #window-1)
            local y_seq = window:sub(2, #window)
            x_sequences[#x_sequences + 1] = x_seq
            y_sequences[#y_sequences + 1] = y_seq
        end
    end

    return x_sequences, y_sequences
end


function MinibatchLoader.sequences_to_tensor(input_sequences, seq_length, vocab_mapping)
    print('putting data into tensor...')
--    local data_tensor = torch.DoubleTensor(#input_sequences, seq_length)
    local data_tensor = torch.ByteTensor(#input_sequences, seq_length)

    for seq_id=1, #input_sequences do
        for char_id=1, seq_length do
            local current_char = input_sequences[seq_id]:sub(char_id, char_id) -- lua has no string indexing using []
--            local current_index = seq_id*seq_length + char_id
--            data_tensor[current_index] = vocab_mapping[current_char]
            data_tensor[seq_id][char_id] = vocab_mapping[current_char]
        end
    end

    return data_tensor
end


function MinibatchLoader.get_table_len(mytable)
    local count = 0
    for _ in pairs(mytable) do
        count = count + 1
    end
    return count
end


function MinibatchLoader.trim_tensor(data, items_num_in_batch)
    -- cut off the end so that it divides evenly
    local saved_dim = data:size(2)
    local total_items_num = data:size(1) * data:size(2)

    if total_items_num % items_num_in_batch ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        local trimmed_items_num = items_num_in_batch * math.floor(total_items_num / items_num_in_batch)
        data = data:view(-1, 1):sub(1,trimmed_items_num):view(-1, saved_dim)
    end

    return data
end

return MinibatchLoader

