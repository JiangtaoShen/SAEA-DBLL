function [OffDec,OffVel] = ReproductionOperator(Problem, PopObj, PopDec, PopVel, B, V, theta)

[N,D] = size(PopDec);
[Winner,WinnerV] = EnvironmentalSelection(PopObj,V,theta);
Loser = setdiff(1:N,Winner);
NL = N - length(Winner);

LoserObj  = PopObj(Loser,:);
WinnerObj = PopObj(Winner,:);
LoserDec  = PopDec(Loser,:);
WinnerDec = PopDec(Winner,:);
LoserVel  = PopVel(Loser,:);
WinnerVel = PopVel(Winner,:);

LoserObj = LoserObj - repmat(min(PopObj,[],1),NL,1);
Angle = acos(1-pdist2(LoserObj,V,'cosine'));
[~,associate] = min(Angle,[],2);
TempWinnerDec = zeros(NL,D);
for i = 1:NL
    [~,index] = ismember(B(associate(i),:),WinnerV);
    index(index == 0) = [];
    if isempty(index)
        IDX = randperm(length(Winner),1);
    else
        IDX = index(randperm(length(index),1));
    end
    TempWinnerDec(i,:) = WinnerDec(IDX,:);
end

r1     = repmat(rand(NL,1),1,D);
r2     = repmat(rand(NL,1),1,D);
OffVel = r1.*LoserVel + r2.*(TempWinnerDec-LoserDec);
OffDec = LoserDec + OffVel + r1.*(OffVel-LoserVel);
OffDec = [OffDec;WinnerDec];
OffVel = [OffVel;WinnerVel];

% Mutation
Lower  = repmat(Problem.lower,N,1);
Upper  = repmat(Problem.upper,N,1);
disM   = 20;
Site   = rand(N,D) < 1/D;
mu     = rand(N,D);
temp   = Site & mu<=0.5;
OffDec       = max(min(OffDec,Upper),Lower);
OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
    (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
temp  = Site & mu>0.5;
OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
    (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

end

