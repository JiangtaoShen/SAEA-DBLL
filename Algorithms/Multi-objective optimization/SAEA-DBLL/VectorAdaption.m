function V = VectorAdaption(V0,PopObj,k)

V = V0.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V0,1),1);
[~,active] = NoActive(PopObj,V);
Va = V(active,:);
if length(active) > k
    Vindex = [];
    [IDX,center]   = kmeans(Va,k);
    for i = unique(IDX)'
        current = find(IDX == i);
        if length(current) == 1
            idx = current;
        else
            Vc = Va(current,:);
            Angle  = acos(1-pdist2(Vc,center(i,:),'cosine'));
            [~,index] = min(Angle);
            idx = current(index);
        end
        Vindex = [Vindex;idx];
    end
    V = Va(Vindex,:);
else
    V = Va;
end

end

