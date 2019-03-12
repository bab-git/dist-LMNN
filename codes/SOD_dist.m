function res=SOD_dist(Dist_perm,a,b);
% (dist(a,b))*(dist(a,b))'

% Computes and sums over all outer products of the columns in Dist_perm.
%
% equivalent to:
%
% res=zeros(size(x,1));
% for i=1:n
%   res=res+x(:,a(i))*x(:,b(i))';
% end;
%
% ********************************************
[D,N,~]=size(Dist_perm);

B=round(2500/D^2*1000000);
res=zeros(D^2,1);
for i=1:B:length(a)
    BB=min(B,length(a)-i);
    
    aa=a(i:i+BB);
    bb=b(i:i+BB);
    sz=size(Dist_perm);
    indA = sub2ind(sz(2:3), aa, bb);
    d_vec=Dist_perm(:,indA);
    prodct=(d_vec)*(d_vec)';
    res=res+vec(prodct);
    if(i>1)   fprintf('.');end;
end;
res=mat(res);

