function PopNew = SampleSelection(PopDec,PopObj,V,mu,theta)

[NVa,va]  = NoActive(PopObj,V);
NCluster  = min(mu,size(V,1)-NVa);
Va        = V(va,:);
[IDX,~]   = kmeans(Va,NCluster);

PopObj = PopObj - repmat(min(PopObj,[],1),size(PopObj,1),1);
cosine = 1 - pdist2(Va,Va,'cosine');
cosine(logical(eye(length(cosine)))) = 0;
gamma  = min(acos(cosine),[],2);
Angle  = acos(1-pdist2(PopObj,Va,'cosine'));
[~,associate] = min(Angle,[],2);

% Next = zeros(1,NV);
APD_S  = ones(size(PopObj,1),1);
for i = unique(associate)'
    current1 = find(associate==i);
    if ~isempty(current1)
        % Calculate the APD value of each solution
        APD = (1+size(PopObj,2)*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
        % Select the one with the minimum APD value
        APD_S(current1,:) = APD;
    end
end

Cindex = IDX(associate); % Solution to cluster
Next = zeros(NCluster,1);

for i = unique(Cindex)'
    solution_Best = [];
    current = find(Cindex==i);
    t = unique(associate(current));
    for j = 1:size(t,1)
        currentS = find(associate==t(j));
        [~,id] = min(APD_S(currentS,:),[],1);
        solution_Best = [solution_Best;currentS(id)];
    end
    [~,best] = min(APD_S(solution_Best,:),[],1);
    Next(i)     = solution_Best(best);
end
index  = Next(Next~=0);
PopNew = PopDec(index,:);

end