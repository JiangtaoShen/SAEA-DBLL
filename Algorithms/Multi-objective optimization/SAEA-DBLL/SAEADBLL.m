classdef SAEADBLL< ALGORITHM
    % <multi/many> <real/integer> <expensive>
    % Surrogate-assisted RVEA
    
    %------------------------------- Reference --------------------------------
    %
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
            %% Parameter settings
            T = 3;
            K = 2;
            mu = 5; wmax = 20; alpha = 2;
            if Problem.M == 10
                NV = 55;
            else
                NV = 50;
            end
            %% Initialization
            V0 = UniformPoint(NV,Problem.M);
            NI    = Problem.D + 50;
            P     = UniformPoint(NI,Problem.D,'Latin');
            A2    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            A1    = A2;
            Model = cell(1,Problem.M);
            V     = V0;
            Ve = V(randperm(size(V,1),ceil(NV/K)),:);
            %% Optimization
            while Algorithm.NotTerminated(A2)
                theta = (Problem.FE/Problem.maxFE)^alpha;
                Ne = min([size(Ve,1);T]);
                B      = pdist2(Ve, Ve);
                [~, B] = sort(B, 2);
                B      = B(:, 1 : Ne);
                A1Dec = A1.decs;
                A1Obj = A1.objs;
                for i = 1 : Problem.M
                    [mS, mY] = dsmerge(A1Dec, A1Obj(:,i));
                    model = rbf_build(mS,mY);
                    Model{i}   = model;
                end
                PopDec = A1Dec;
                PopObj = A1Obj;
                PopVel = zeros(length(A1),Problem.D);
                Ns = [];
                w = 1;
                % Inner loop
                while w <= wmax
                    [OffDec, OffVel] = ReproductionOperator(Problem, PopObj, PopDec, PopVel, B, Ve, theta);
                    PopDec = [PopDec;OffDec];
                    PopVel = [PopVel;OffVel];
                    [N,~]  = size(PopDec);
                    PopObj = zeros(N,Problem.M);
                    for i = 1: N
                        for j = 1 : Problem.M
                            PopObj(i,j) = rbf_predict(Model{j}, mS, PopDec(i,:));
                        end
                    end
                    [index,~] = EnvironmentalSelection(PopObj,V,theta);
                    Ns = [Ns;length(index)];
                    PopDec = PopDec(index,:);
                    PopObj = PopObj(index,:);
                    PopVel = PopVel(index,:);
                    w = w + 1;
                end
                V = V0.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V0,1),1);
                Ve = VectorAdaption(V0,PopObj,ceil(mean(Ns)/K));
                PopNew = SampleSelection(PopDec,PopObj,V,mu,theta);
                % Update the archive
                if ~isempty(PopNew)
                    if size(PopNew,1) > Problem.maxFE - Problem.FE
                        PopNew = PopNew(1:Problem.maxFE - Problem.FE,:);
                    end
                    New       = Problem.Evaluation(PopNew);
                    A2        = [A2,New];
                    A1        = UpdataArchive(A1,New);
                end
                FN = NDSort(A2.objs,inf);
                A2 = A2(FN == 1);
            end
        end
    end
end