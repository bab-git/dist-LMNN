function [train,test]=makesplits_CV(y,split,splits,classsplit,k,dummy)
%  function [train,test]=makesplits(y,split,splits,classsplit)
%
% SPLITS "y" into "splits" sets with a "split" ratio.
% if classsplit==1 then it takes a "split" fraction from each class
%

if(split==1)
    CVP_profile=cvpartition(y,'Resubstitution');
    train=find(CVP_profile.training);    
    test=[];
    return;
end;


if(split==0)
    test=randperm(length(y));
    train=[];
    return;
end;


if(nargin<4)
    classplit=0;
end;
if(nargin<5)
    k=1;
end;

n=length(y);
if(minclass(y,1:length(y))<k || split*length(y)/length(unique(y))<k)
    fprintf('K:%i split:%f n:%i\n',k,split,length(y));
    keyboard;
    error('Cannot sub-sample splits! Reduce number of neighbors.');
end;


if(classsplit)
    un=unique(y);
    for i=1:splits
        trsplit=[];
        tesplit=[];
        while(minclass(y,trsplit)<k)
            for j=1:length(un)
                ii=find(y==un(j));
                try
                    %                     CVP_profile=cvpartition(ii,'holdout',floor(length(ii)*(1-split)));
                    w=warning('off', 'all');
                    CVP_profile=cvpartition(length(ii),'holdout',ceil(length(ii)*(1-split)));
                    w=warning('on', 'all');
                catch err
                    save temp_err
                end
                trsplit=[trsplit ii(find(CVP_profile.training))];
                tesplit=[tesplit ii(find(CVP_profile.test))];
                                
            end;
        end;
        train(i,:)=trsplit(randperm(length(trsplit)));
        test(i,:)=tesplit(randperm(length(tesplit)));;
    end;
else
    for i=1:splits
        trsplit=[];
        tesplit=[];
        ic=1;
        while(minclass(y,trsplit)<k)
            CVP_profile=cvpartition(y,'holdout',floor(n*(1-split)));
            tesplit=find(CVP_profile.test);
            trsplit=find(CVP_profile.training);            
            ic=ic+1;
            if ic>20
                error('low number of sample in each class/ or high number of neighbours')
            end
        end;
        train(i,:)=trsplit;
        test(i,:)=tesplit;
    end;
end;




function [m]=minclass(y,ind)

un=unique(y);
m=inf;
for i=1:length(un)
    m=min(sum(y(ind)==un(i)),m);
end;

