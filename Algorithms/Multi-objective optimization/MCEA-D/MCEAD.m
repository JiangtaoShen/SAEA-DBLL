classdef MCEAD < ALGORITHM
    % <multi/many> <real/integer> <expensive>
    % Multiple classifiers-assisted evolutionary algorithm based on decomposition
    % delta  --- 0.9 --- The probability of choosing parents locally
    % nr     ---   2 --- Maximum number of solutions replaced by each offspring
    % Rmax   ---  10 --- Maximum repeat time of offspring generation
    
    %------------------------------- Reference --------------------------------
    % T. Sonoda and M. Nakata, Multiple classifiers-assisted evolutionary
    % algorithm based on decomposition for high-dimensional multi-objective
    % problems, IEEE Transactions on Evolutionary Computation, 2022.
    %--------------------------------------------------------------------------
    
    % This function is written by Masaya Nakata
    
    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [delta, nr, R_max] = Algorithm.ParameterSet(0.9, 2, 10);            
            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            
            %% Detect the neighbours of each solution
            T      = ceil(Problem.N / 10);
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1 : T);
            
            %% Initialize population
            NI = Problem.D + 50;
            PopDec     = UniformPoint(NI, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper - Problem.lower, NI, 1) .* PopDec + repmat(Problem.lower, NI, 1));
            A          = Population;
            Z          = min(Population.objs, [], 1);
            A2 = A;
            %% Define SVM
            svm_list = SVM(Problem);
            
            %% Optimization
            while Algorithm.NotTerminated(A2)
                % For each sub-problem
                for i = 1 : Problem.N
                    %% Model-construction
                    svm_list(i) = svm_list(i).ModelConstruction(A, B(i, :), W, Z);
                    %% Choose the parents
                    if rand < delta
                        P = B(i, randperm(end));
                    else
                        P = randperm(Problem.N);
                    end
                    %% Solution-generation
                    y_i = SolutionGeneration(Problem, Population, P, svm_list(i), R_max, i);
                    %% Evaluate offspring
                    y_i = Problem.Evaluation(y_i);
                    %% Update the reference point
                    Z = min(Z, y_i.obj);
                    %% Update population and archive
                    g_old = max(abs(Population(P).objs - repmat(Z, length(P), 1)) .* W(P, :), [], 2);
                    g_new = max(repmat(abs(y_i.obj - Z), length(P), 1) .* W(P, :), [], 2);
                    Population(P(find(g_old >= g_new, nr))) = y_i;
                    A = [A, y_i];
                    A2 = [A2, y_i];
                    FN = NDSort(A2.objs,inf);
                    A2 = A2(FN == 1);
                    %% Check termination criteria
                    Algorithm.NotTerminated(A2);
                end
            end
        end
    end
end