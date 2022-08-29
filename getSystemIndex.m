%%   HD 单个用户的海明距离,衡量用户间推荐的不同程度。即推荐列表的不同程度. 指的是推荐列表L之间的物品
%     Instra 内部用户推荐的不同程度
%     pop    用户推荐的流行程度
% 输入： 1.训练集  2.预测集里浏览的物品列表 3,推荐列表的长度  用户个数 物品个数

function  [h Instra Pop allh]=getSystemIndex(traindata,itemdu,pre,L,UserExistBoth,m,n)
    pre=pre(UserExistBoth,:);
    userSize=size(pre,1);

    %% 海明距离
    %获取推荐列表前L个物品
    for i=1:userSize
        temp=pre(i,:);
        [~,C]=sort(pre(i,:));
        temp(C(1:n-L))=0;
        pre(i,:)=temp;
    end
     pre(pre>0)=1;
     clear temp C;

    %对pre的两两用户之间求海明距离
     p=1-(pre*pre'./L);
     p=p-p.*eye(userSize);
     allh = sum(p,1);
     h=sum(sum(p,2),1);
     h=h/(userSize*(userSize-1));
     clear p;
     %% 内部距离 即单个用户被推荐的商品之间的相似性
     Instra=0;
     ItemFenmu=sqrt(itemdu'*itemdu);
     ItemSim=traindata'*traindata./ItemFenmu;
     clear ItemFenmu;
     ItemSim(isinf(ItemSim))=0;
     ItemSim(isnan(ItemSim))=0;
     %每个物品对有多少用户共同购买
     comm = pre'*pre.*ItemSim;
     Instra1 = (sum(sum(comm,2),1)-trace(comm))/2;
     clear  ItemSim s;
     Instra=Instra1*2/(L*(L-1)*userSize);

     %% 度量推荐新颖性，即商品的平均度
     Pop = sum(pre*(itemdu'),1)/(userSize*L);
     clear pre;
end
 
 