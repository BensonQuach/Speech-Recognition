addpath(genpath('/h/u6/g0/00/g0quachb/Desktop/A3-401/FullBNT-1.0.4'));
%---- Variables----
  format long
  
  d = 14;
  M = 8;
  Q = 3;
  N = 30;
  saved_result_name = 'trained_hmms.mat';
  
  initType = 'kmeans';
  max_iter = 10;
  dir_path = '/u/cs401/speechdata/Training/';
%------------------

speaker_count = 0;

phn_struct = struct();
data_dir = dir([dir_path]);
length_of_DD = length(data_dir);

for k=1:length_of_DD
    if (data_dir(k).isdir) & isempty(strfind(data_dir(k).name, '.'))
        curr_speaker = data_dir(k).name;
        dir_path_speaker = [dir_path curr_speaker '/'];
        data_speaker = dir([dir_path_speaker, './*mfcc']);
        length_of_DD_spker = length(data_speaker);
        
        for f=1:length_of_DD_spker
            MFCC_Matrix = [];
          
            f_name = data_speaker(f).name;
            MFCC_Matrix = textread([dir_path_speaker, f_name]);
            MFCC_Matrix = MFCC_Matrix(:, 1:d);

            limit = size(MFCC_Matrix, 1);
            phn_file = [f_name(1:findstr(f_name, '.')) 'phn'];
            
            data_phn = textread([dir_path_speaker, phn_file], '%s', 'delimiter', '\n');
            len_of_dataphn = length(data_phn);
            for b=1:len_of_dataphn
                In = sscanf(data_phn{b}, '%d %d %s');
                phone = char(In(3:end)');
                
                if strcmp(phone, 'h#')
                    phone = 'sil';
                end
                
                START = (In(1) / 128) + 1;
                END = min(limit, (In(2) / 128) + 1);
                
                sliced_mfcc = MFCC_Matrix(START:END, 1:d);
                
                if isfield(phn_struct, phone)
                    phn_struct.(phone){length(phn_struct.(phone)) + 1} = sliced_mfcc';
                else
                    phn_struct.(phone) = {sliced_mfcc'};
                end
            end
        end
        
        speaker_count = speaker_count + 1;
        if speaker_count == N
            break
        end
        
    end
end

%save('phn_struct.mat', 'phn_struct', '-mat');

phn_len = length(fields(phn_struct));
phn_fields = fieldnames(phn_struct);

trained_hmms = struct();
for k=1:phn_len
    c_field = phn_fields{k};
    data_len = length(phn_struct.(c_field));
    data = phn_struct.(c_field);
    HMM = initHMM( data, M, Q, initType );
    [HMM, LL] = trainHMM( HMM, data, max_iter ); 
    trained_hmms.(c_field) = HMM;
end

save(saved_result_name, 'trained_hmms', '-mat');

trained_hmms
        