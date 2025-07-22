function  A1 = UpdataArchive(A1,New)

All       = [A1.decs;New.decs];
[~,index] = unique(All,'rows');
ALL       = [A1,New];
A1     = ALL(index);

end