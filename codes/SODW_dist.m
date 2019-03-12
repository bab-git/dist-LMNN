function res=SODW_dist(Dist_perm,a,b,w);
% (dist(a,b))*(dist(a,b))'

% Computes and sums over all outer products of the columns in x.
%
% equivalent to:
%
% res=zeros(size(x,1));
% for i=1:n
%   res=res+x(:,a(i))*diag(W)*x(:,b(i))';
% end;
%
% ********************************************

[D,N,~]=size(Dist_perm);
B=round(2500/D^2*1000000);
res=zeros(D^2,1);
% =======================
aa=a(:);
bb=b(:);
sz=size(Dist_perm);
indA = sub2ind(sz(2:3), aa, bb);
d_vec=Dist_perm(:,indA);
dgw=diag(w);
prodct=(d_vec)*dgw*(d_vec)';
res=res+vec(prodct);
res=mat(res);

