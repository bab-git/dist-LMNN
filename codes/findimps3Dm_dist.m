function ind=findimps3Dm_dist(Dist_L,X1,X2,t1,t2);
% Takes two sets of vectors as input with accompaning thresholds. 
% Finds all indices (i,j) of columns in X1 and X2 such the L2 squared
% distance between vectors X1(:,i) and X2(:,j) is less-equal than
% t1(i) or t2(j).  
%
% equivalent to: 
%
%      Dist=Dist_L(X1,X2);  % computes L2-distance matrix
%      imp1=find(Dist<repmat(t1,N1,1))';
%      imp2=find(Dist<repmat(t2,N1,1))';
%      [a,b]=ind2sub([N1,N2],[imp1;imp2]);
%      ind=[b;a];
%      
%
%
% ********************************************

N1=size(X1,2);
N2=size(X2,2);
if(size(t1,1)==1) t1=t1.';end;
if(size(t2,2)==1) t2=t2.';end;

Dist=Dist_L(X1,X2);

imp1=find(bsxfun(@lt,Dist,t2))';imp1=imp1(:)';
imp2=find(bsxfun(@lt,Dist,t1))';imp2=imp2(:)';
  
[a,b]=ind2sub([N1,N2],[imp1 imp2]);
ind=[a;b];
ind=unique(ind','rows')';
	  