classdef REMO < ALGORITHM
    % <multi/many> <real> <expensive>
    % Expensive multiobjective optimization by relation learning and prediction
    % k    ---    6 --- Number of reference solutions
    % gmax --- 3000 --- Number of solutions evaluated by surrogate model
    
    %------------------------------- Reference --------------------------------
    % H. Hao, A. Zhou, H. Qian, and H. Zhang, Expensive multiobjective
    % optimization by relation learning and prediction, IEEE Transactions on
    % Evolutionary Computation, 2022.
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    methods
        function main(Algorithm,Problem)
            %% Parameterr setting
            [k,gmax] = Algorithm.ParameterSet(6,3000);
            
            %% Initalize the population by Latin hypercube sampling
            N = Problem.D + 50;
            PopDec     = UniformPoint(N,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*PopDec+repmat(Problem.lower,N,1));
            Archive    = Population;
            A2 = Archive;
            %% Optimization
            while Algorithm.NotTerminated(A2)
                % Select reference solutions and preprocess the data
                Ref       = RefSelect(Population,k);
                Input     = Population.decs;
                Catalog   = GetOutput_PBI(Population.objs,Ref.objs);
                [XXs,YYs] = GetRelationPairs(Input,Catalog);
                [TrainIn,TrainOut,TestIn,TestOut] = DataProcess(XXs,YYs);
                xDim = size(TrainIn,2);
                
                % Train relation model
                [TrainIn_nor,TrainIn_struct] = mapminmax(TrainIn');
                TrainIn_nor     = TrainIn_nor';
                TrainOut_onehot = onehotconv(TrainOut,1);
                net = patternnet([ceil(xDim*1.5),xDim*1,ceil(xDim/2)]);
                net.trainParam.showWindow =0;
                net        = train(net,TrainIn_nor',TrainOut_onehot');
                TestIn_nor = mapminmax('apply',TestIn',TrainIn_struct)';
                TestPre    = onehotconv(net(TestIn_nor')',2);
                p_err      = sum(TestPre ~= TestOut)/size(TestPre,1);
                Smodel.X   = Input;
                Smodel.Y   = Catalog;
                Smodel.mp_struct = TrainIn_struct;
                Smodel.net       = net;
                Smodel.p_err     = p_err;
                Next = RSurrogateAssistedSelection(Problem,Ref,Population.decs,gmax,Smodel);
                if ~isempty(Next)
                    if size(Next,1) > Problem.maxFE - Problem.FE
                        Next = Next(1:Problem.maxFE - Problem.FE,:);
                    end
                    New       = Problem.Evaluation(Next);
                    Archive = [Archive,New];
                    A2 = Archive;
                end
                Population = RefSelect(Archive,Problem.N);
                FN = NDSort(A2.objs,inf);
                A2 = A2(FN == 1);
            end
        end
    end
end