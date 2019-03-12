function [accuracy, labels ,score,cost]=knn_eval_dis(L,alldistances,r_tr,yTr,r_te,yTe,Knn)
[N,~,Nf]=size(alldistances);

tempA=permute(alldistances,[3,1,2]);
tempB=reshape(tempA,Nf,[]);
tempC=L*tempB;
tempD=sum(tempC.*tempC,1);
Dist_L=reshape(tempD,N,N);

xt=r_tr';
xe=r_te';

dist_fnc = @(x,Z) Dist_L(Z,x);
knnmodel=fitcknn(xt, yTr,'NumNeighbors',Knn,'Distance',@(x,Z)dist_fnc(x,Z)); 
[labels,score,cost] = predict(knnmodel,xe);    
accuracy=(1-sum(labels~=yTe)/length(yTe))*100;
end