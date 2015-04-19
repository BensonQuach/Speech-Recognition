function [SE IE DE LEV_DIST] = Levenshtein(hypothesis,annotation_dir)

format long;
%hypothesis = '/u/cs401/speechdata/Testing/hypotheses.txt'
%annotation_dir = '/u/cs401/speechdata/Testing/'
%[SE IE DE LEV_DIST] = Levenshtein(hypothesis,annotation_dir)

% Input:
%	hypothesis: The path to file containing the the recognition hypotheses
%	annotation_dir: The path to directory containing the annotations
%			(Ex. the Testing dir containing all the *.txt files)
% Outputs:
%	SE: proportion of substitution errors over all the hypotheses
%	IE: proportion of insertion errors over all the hypotheses
%	DE: proportion of deletion errors over all the hypotheses
%	LEV_DIST: proportion of overall error in all hypotheses

SE = 0;
IE = 0;
DE = 0;
LEV_DIST = 0;

total_REF_LEN = 0;

data_hypo = textread(hypothesis, '%s', 'delimiter', '\n');
len_of_data_hypo = length(data_hypo);

for b=1:len_of_data_hypo
    data_annot = textread([annotation_dir 'unkn_' num2str(b) '.txt'], '%s', 'delimiter', '\n');
    anotSentence = regexprep(data_annot{1}, '[0-9]+ [0-9]+ (.*)', '$1');
    REF = strsplit(' ', anotSentence);
    total_REF_LEN = total_REF_LEN + length(REF);
end

for b=1:len_of_data_hypo
    
    hypSentence = regexprep(data_hypo{b}, '[0-9]+ [0-9]+ (.*)', '$1');
    data_annot = textread([annotation_dir 'unkn_' num2str(b) '.txt'], '%s', 'delimiter', '\n');
    anotSentence = regexprep(data_annot{1}, '[0-9]+ [0-9]+ (.*)', '$1');
    
    HYP = strsplit(' ', hypSentence);
    REF = strsplit(' ', anotSentence);

    m = length(HYP) + 1;
    n = length(REF) + 1;
    R = cell(n + 1, m + 1); % Matrix of disntances
    B = cell(n + 1, m + 1); % Backtracking matrix
    
    len_row = n + 1;
    len_col = m + 1; 
    
    for i=1:len_row
        for j=1:len_col
            if (i == 1 || j == 1)
                R{i, j} = Inf;
            end
        end
    end
    
    R{1, 1} = 0;
    
    for i=2:n
        for j=2:m
            del = R{i-1, j} + 1;
            if strcmp(REF{i - 1}, HYP{j - 1})
                val = 0;
            else
                val = 1;
            end
            sub = R{i-1, j-1} + val;
            ins = R{i, j-1} + 1;
            R{i,j} = min(del, min(sub, ins));
            if R{i,j} == del
                B{i,j} = 'up';
            elseif R{i,j} == ins
                B{i,j} = 'left';
            else 
                B{i,j} = 'up-left';
            end
        end
    end

    total_REF_LEN = total_REF_LEN + n;
    LEV_DIST = LEV_DIST + ((100 * (R{n, m} / n)) * (length(REF) / total_REF_LEN));
    
    total_SE = 0;
    total_IE = 0;
    total_DE = 0;
    i = len_row;
    j = len_col;
    
    while i > 1 && j > 1
        if strcmp(B{i, j}, 'up')
            total_DE = total_DE + 1;
            i = i - 1;
        elseif strcmp(B{i, j}, 'left')
            total_IE = total_IE + 1;
            j = j - 1;
        else 
            if strcmp(B{i, j}, 'up-left') && not(strcmp(REF{i - 1}, HYP{j - 1}))
                total_SE = total_SE + 1;
            end
            i = i - 1;
            j = j - 1;
        end
    end

    SE = SE + total_SE * (length(REF) / total_REF_LEN);
    IE = IE + total_IE * (length(REF) / total_REF_LEN);
    DE = DE + total_DE * (length(REF) / total_REF_LEN);
    
    disp(['sentence ' num2str(b) ', DE: ' num2str(total_DE) ', IE: ' num2str(total_IE) ', SE: ' num2str(total_SE) ', LEV_DIST: ' num2str(100 * (R{n, m} / n))]);
end

