addpath(genpath('/h/u6/g0/00/g0quachb/Desktop/A3-401/FullBNT-1.0.4'));

load trained_hmms.mat

%---- Variables----
  format long
  
  d = 14;
  
  dir_path = '/u/cs401/speechdata/Testing/';
%------------------

phn_struct_test = struct();

% Build entire MFCC of Testing dir.
mfcc_files = dir([dir_path, './*mfcc']);
length_data_mfcc = length(mfcc_files);
for f=1:length_data_mfcc

    % For every unkn_*.mfcc (called it f_name)
    %   [1] construct the entire MFCC_Matrix
    f_name = mfcc_files(f).name;
    if not(mfcc_files(f).isdir)

        MFCC_Matrix = textread([dir_path, f_name]);
        MFCC_Matrix = MFCC_Matrix(:, 1:d);
        
        % Determine the length of this unkn_*.mfcc length.
        limit = size(MFCC_Matrix, 1);
        % Get respective phoneme file (called phn_file)
        phn_file = [f_name(1:findstr(f_name, '.')) 'phn'];

        % Read the entire data of the phoneme file.
        %   Read it into a struct that will contain all the MFCC slices for
        %   each phoneme.
        data_phn = textread([dir_path, phn_file], '%s', 'delimiter', '\n');
        len_of_dataphn = length(data_phn);
        for b=1:len_of_dataphn

            % '%s' contains the CORRECT answer to the sliced MFCC phoneme.
            %   represented slicing in %d to %d of MFCC_Matrix
            In = sscanf(data_phn{b}, '%d %d %s');
            phone = char(In(3:end)');

            if strcmp(phone, 'h#')
                phone = 'sil';
            end

            START = (In(1) / 128) + 1;
            END = min(limit, (In(2) / 128) + 1);

            % slice the MFCC_Matrix for it
            %   (recall: 'phone' contains CORRECT phoneme)
            sliced_mfcc = MFCC_Matrix(START:END, 1:d);

            if isfield(phn_struct_test, phone)
                phn_struct_test.(phone){length(phn_struct_test.(phone)) + 1} = sliced_mfcc';
            else
                phn_struct_test.(phone) = {sliced_mfcc'};
            end
        end

    end
end

%save('phn_struct_test.mat', 'phn_struct_test', '-mat');

phn_len = length(fields(phn_struct_test));
phn_fields = fieldnames(phn_struct_test);

disp('Moving to Final Phase!')

num_of_matrices = 0;
overall_accuracy = 0;

% Recall: 'phn_struct_test' contains phoneme fields which are the CORRECT
%   answer to the sliced matrices provided in the field.  Those matrices
%   are data passed in loglikHMM to determine a LL value.
%   (ie. each sliced matrix has a LL value we get that matrix with
%       data{index}.)

% Furthermore phn_struct_test.(CORRECT phoneme){1} is essentially 1 line
%   in the unkn_*.phn file.
LL_phn_struct = struct();
for k=1:phn_len
    disp(['Entering phoneme ' num2str(k) '!'])
    % 'c_field' contains the correct phoneme.
    c_field = phn_fields{k};
    % determine the number of sliced mfcc matrices.
    data_len = length(phn_struct_test.(c_field));
    data = phn_struct_test.(c_field);
    
    % For each sliced matrix, run through each phoneme in trained_hmms
    %   (ie. fields) get the max(LLs) w/ respective phoneme and check
    %   if it's equivalent to c_field.
    hmms_fields = fields(trained_hmms);
    hmms_len_fields = length(hmms_fields);
    correct = 0;
    % This is going through each sliced mfcc matrix.
    disp('  Going through sliced MFCC Matrices!')
    for b=1:data_len
        % Gather all LL data w/ resp. phoneme. for that one sliced mfcc.
        LL_vector = [];
        hmms_phn_vector = {};
        for c=1:hmms_len_fields
            sub_c_field = hmms_fields{c};
            LL = loglikHMM(trained_hmms.(sub_c_field), data{b});
            LL_vector = [LL_vector LL];
            hmms_phn_vector{c} = sub_c_field;
        end
        % Out of all the phonemes. Was the maximum == to the correct
        % phoneme?
        max_ind = find(LL_vector==max(LL_vector));
        if strcmp(hmms_phn_vector{max_ind(1)}, c_field)
            correct = correct + 1;
            disp('CORRECT!')
        else
            disp('INCORRECT!')
        end
    end

    num_of_matrices = num_of_matrices + data_len;
    overall_accuracy = overall_accuracy + correct;
    disp([num2str(k/phn_len) '% complete.'])
    disp([c_field ': ' num2str(correct) '/' num2str(data_len) ' = ' num2str(correct / data_len)])
    LL_phn_struct.(c_field) = [num2str(correct) '/' num2str(data_len) ' = ' num2str(correct / data_len)];
end

overall_accuracy = 100 * (overall_accuracy / num_of_matrices);
save('overall_accuracy.mat', 'overall_accuracy', '-mat');
save('LL_phn_struct.mat', 'LL_phn_struct', '-mat');

LL_phn_struct
overall_accuracy
