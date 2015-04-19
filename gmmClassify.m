%---- Variables----

d = 14;
M = 8;
epsilon = 0.001;

MAX_ITER = 20;

dir_path_test = '/u/cs401/speechdata/Testing/';
dir_path_train = '/u/cs401/speechdata/Training/';
%------------------

% Create the MFCC Matrix of the Testing directory.
MFCC_collection = [];
MFCC_count = 0;

data_dir = dir([dir_path_test, './*mfcc']);
length_of_DD = length(data_dir);

for k=1:length_of_DD
    MFCC = [];
    if not(data_dir(k).isdir)
        curr_file = data_dir(k).name;
        data = textread([dir_path_test, curr_file], '%s', 'delimiter', '\n');

        len_of_curr_mfcc = length(data);

        for b=1:len_of_curr_mfcc
            MFCC = [MFCC; str2num(char(data(b)))];
        end
        MFCC_count = MFCC_count + 1;
        
        MFCC = MFCC(:, 1:d);

        utter.utter_fname = curr_file(1:strfind(curr_file, '.'));
        utter.utter_MFCC = MFCC;
        MFCC_collection = [MFCC_collection utter];
    end
end

gmms = gmmTrain(dir_path_train, MAX_ITER, epsilon, M);

% Reconstruct respective theta for each speaker
num_of_speakers = length(gmms);
theta_collection = [];
for s=1:num_of_speakers
    theta = [];
    for m=1:M
        index = ['m_' int2str(m)];
        theta.(index).omega = gmms(s).weights(m);
        theta.(index).mu = gmms(s).means(:, m)';
        theta.(index).sigma = diag(gmms(s).cov(:, :, m))';
    end
    % Name given also to retain for output to file.
    theta.name = gmms(s).name;
    
    theta_collection = [theta_collection theta];
end

numOfUtterances = length(MFCC_collection);
for u=1:numOfUtterances
    
    % Find top 5 speakers for this utterance. (MFCC)
    MFCC = MFCC_collection(u).utter_MFCC;
    utterance_name = MFCC_collection(u).utter_fname;
    loglik_candidates = [];
    name_of_candidates = {};
    
    for s=1:num_of_speakers
        
        % Get the current theta in consideration for MFCC above.
        theta = theta_collection(s);

        %============= CALCULATE LOGLIKLIHOOD ================
        numOfTrainingPoints = length(MFCC);
        LOG_b_matrix = [];

        % Perform Likelihood computation.
        for m=1:M
            index = ['m_' int2str(m)];

            % All terms are 1x14 matrices
            term1 = -0.5 * theta.(index).sigma.^(-1);
            term2 = theta.(index).sigma.^(-1) .* theta.(index).mu;
            term3 = -0.5 * (theta.(index).sigma.^(-1) .* theta.(index).mu.^(2));
            log_section = ((d/2) * log(2*pi)) + ((1/2) * log(prod(theta.(index).sigma)));
            
            LOG_b_vect = [];
            for t=1:numOfTrainingPoints
                trainingPoint = MFCC(t, 1:d);
                LOG_b_m_xt = sum((term1 .* (trainingPoint.^2)) + (term2 .* trainingPoint) + term3) - log_section;
                LOG_b_vect = [LOG_b_vect LOG_b_m_xt];
            end
            LOG_b_matrix = [LOG_b_matrix; LOG_b_vect];

        end
        b_matrix = exp(LOG_b_matrix);

        % p_theta is Probability of xt in the GMM
        p_theta = zeros(1, size(b_matrix, 2));
        for m=1:M
            index = ['m_' int2str(m)];
            p_theta = p_theta + theta.(index).omega * b_matrix(m, :);
        end

        % Calculate loglikelihood.
        L = sum(log(p_theta));

        %======================================================
        loglik_candidates = [loglik_candidates L];
        name_of_candidates{s} = theta.name;
    end

    [sorted_val sorted_ind] = sort(loglik_candidates(:), 'descend');

    file_lik = fopen([utterance_name 'lik'], 'w');
    fprintf(file_lik, [name_of_candidates{sorted_ind(1)} ' ' num2str(sorted_val(1)) '\n']);
    fprintf(file_lik, [name_of_candidates{sorted_ind(2)} ' ' num2str(sorted_val(2)) '\n']);
    fprintf(file_lik, [name_of_candidates{sorted_ind(3)} ' ' num2str(sorted_val(3)) '\n']);
    fprintf(file_lik, [name_of_candidates{sorted_ind(4)} ' ' num2str(sorted_val(4)) '\n']);
    fprintf(file_lik, [name_of_candidates{sorted_ind(5)} ' ' num2str(sorted_val(5)) '\n']);

end
