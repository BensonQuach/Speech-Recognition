function gmms = gmmTrain(dir_path, MAX_ITER, epsilon, M)

%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of theta weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture

%---- Variables----
  format long
  d = 14;
  N = 30;
%------------------

  speaker_count = 0;

  data_dir = dir([dir_path]);
  length_of_DD = length(data_dir);
  gmms = [];  

  for k=1:length_of_DD
 	
  	if (data_dir(k).isdir) & isempty(strfind(data_dir(k).name, '.'))
		curr_speaker = data_dir(k).name;
		gmm = gmmTrainOne([dir_path '/' curr_speaker '/'], MAX_ITER, epsilon, M, d, curr_speaker);
		gmms = [gmms gmm];
        
        speaker_count = speaker_count + 1;
        if speaker_count == N
            break
        end
	end
  end 

  return
end

% --------------------------------------------------------------------------------
%  Support functions
% --------------------------------------------------------------------------------
function gmm = gmmTrainOne(dir_path, MAX_ITER, epsilon, M, d, name)

  X = readIn(dir_path, d);
  theta = initialize(X, d, M);

  %----- Algorithm provided in Assignment -----
  i = 0;
  prev_L.log_L = -Inf;
  improvement = Inf;
  while i <= MAX_ITER & improvement >= epsilon
	L = ComputeLikelihood(X, theta, d, M);
	theta = UpdateParameters(theta, X, L, M);
	improvement = L.log_L - prev_L.log_L;
	prev_L = L;
	i = i + 1;
  end
  %---------------------------------------------
  
  gmm.name = name;
  gmm.weights = [];
  gmm.means = [];
  gmm.cov = zeros(d, d, 1);
  for m=1:M
	index = ['m_' int2str(m)];
	gmm.weights = [gmm.weights theta.(index).omega];
  	gmm.means = [gmm.means theta.(index).mu'];
 	gmm.cov(:, :, m) = diag(theta.(index).sigma);
  end

  return
end

% Given the training directory of and up to the speaker folder. 
% Construct a MFCC matrix consisting of all the MFCC files within
% that directory. Return that MFCC matrix.
function MFCC_Matrix = readIn(dir_path, d)
 
  MFCC_Matrix = []; 
  data_dir = dir([dir_path, './*mfcc']);
  length_of_DD = length(data_dir);
  
  for k=1:length_of_DD

    if not(data_dir(k).isdir)
        curr_file = data_dir(k).name;
        data = textread([dir_path, curr_file], '%s', 'delimiter', '\n');

        len_of_curr_mfcc = length(data);
        for b=1:len_of_curr_mfcc
            MFCC_Matrix = [MFCC_Matrix; str2num(char(data(b)))];
        end
     end
  end

  MFCC_Matrix = MFCC_Matrix(:, 1:d);
  
  return
end

% Given the MFCC Matrix, initialize the theta struct that will contain
% the omega, mu, and sigma of the respective M index.
function theta = initialize(MFCC, d, M)

  theta = [];
  random_num = randperm(length(MFCC)); 
 
  for k=1:M
	index = ['m_' int2str(k)];
  	mu = MFCC(random_num(k), 1:d);

  	sigma = ones(1, d);
  	omega = 1 / M;
    theta.(index).omega = omega;	
	theta.(index).mu = mu; 
    theta.(index).sigma = sigma; 
  end

  return

end

function L = ComputeLikelihood(MFCC, theta, d, M)

  % d = length(theta),
  % M is the matrix size we're considering for efficiency. 
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
		trainingPoint = MFCC(t, :);
		LOG_b_m_xt = sum((term1 .* (trainingPoint.^2)) + (term2 .* trainingPoint) + term3) - log_section;
		LOG_b_vect = [LOG_b_vect LOG_b_m_xt];
    end

	LOG_b_matrix = [LOG_b_matrix; LOG_b_vect];

  end
  b_matrix = exp(LOG_b_matrix);
  
  % p_theta is Probability of xt in the GMM, USED TO: calculate p_gaussian
  % [1x1]
  p_theta = zeros(1, size(b_matrix, 2));
  for m=1:M
	index = ['m_' int2str(m)];
	p_theta = p_theta + theta.(index).omega * b_matrix(m, :);
  end

  % p_gaussian is Probability of the mth Gaussian, given xt ; P(rm|xt, theta)
  % [8x2726] [MxT]
  p_gaussian = [];
  for m=1:M
	index = ['m_' int2str(m)];
	p_gaussian = [p_gaussian; (theta.(index).omega ./ p_theta) .* b_matrix(m, :)];	
  end

  % Calculate loglikelihood.
  L.log_L = sum(log(p_theta));
  L.p_gaussian = p_gaussian;

  return

end

function theta = UpdateParameters(theta, MFCC, L, M)

  T = length(L.p_gaussian(1, :));

  for m=1:M
  	curr_P_gaus_sum = sum(L.p_gaussian(m, :));
	index = ['m_' int2str(m)];

	% Calculate the numerators.
	mu_numerator = zeros(1, length(MFCC(1, :)));
	sigma_numerator = zeros(1, length(MFCC(1, :)));
	for t=1:T
  		mu_numerator = mu_numerator + (L.p_gaussian(m, t) .* MFCC(t, :));
		sigma_numerator = sigma_numerator + (L.p_gaussian(m, t) .* MFCC(t, :).^2);
  	end

  % Updating OMEGA
  	theta.(index).omega = (1/T) * curr_P_gaus_sum;

  % Updating MU [MFCC is 2726x14]
  	theta.(index).mu = mu_numerator  / curr_P_gaus_sum;

  % Updating SIGMA (covariance which is essentially variance)
	theta.(index).sigma = (sigma_numerator / curr_P_gaus_sum) - theta.(index).mu.^2 ;
  end

end

