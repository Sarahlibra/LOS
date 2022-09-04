%% 实现HHP算法
clear;clc;
%% 读入用户CH2-L2相似度矩阵
userSim = load('Data\\friendfeed\\social_data.txt');
% userSim(:,3) = userSim(:,3)./sum(userSim(:,3));
userSimM = full(spconvert(userSim));
%补全矩阵
[m,n]=size(userSimM);
if m > n
    userSimM(:,n+1:m) = 0;
elseif m < n
    userSimM(m+1:n,:)=0;
end
userSimM = userSimM + userSimM';
userSimM(userSimM~=0)=1;
%% 读入数据，转换成矩阵, 划分训练集，测试集
data = load('Data\\friendfeed\\ratings_data.txt');
data(:,3)=1;
Net = spconvert(data);
direction = 1;
ratioTrain = 0.9;
simulation = 20;
L = 0;
recL = 50;
indexResult=zeros(10000,10);
degree_list = [];
eva_list = [];
gidx = 0;
ddix = 0;
bins = 1;

%friendfeed: active, 30, inactive, 9, cold 3;
%Epinions: active, 54, inactive, 25, cold 13
threshold=25; 
plist = [];
for r = 1:simulation
     [train test] = DivideNet(Net, ratioTrain, direction);

    %将读入的用户  物品 打分 转换为用户*物品的(0,1)二维矩阵
    train_data=full(train);  %训练集用户的资源分配矩阵
    test_data=full(test);
    clear train test;
    train_user_du=sum(train_data,2); %获取用户度
    test_user_du=sum(test_data,2);
    train_item_du = sum(train_data,1);%商品的度
    [m,n]=size(train_data);
    %获取测试集或者训练集没有打分的用户
    index1 = find(train_user_du==0);
    index2 = find(test_user_du==0);
    index_ = unique(union(index1,index2)); 
    clear index1 index2;
    index4 = find(test_user_du~=0);
%     UserExistsBoth = intersect(index3,index4);%%%用于评价指标的计算
    UserExistsBoth = index4;
    clear index4;

    %获取第一步传播前TopN个邻居用户
    I = eye(m);
    AA = train_data * train_data';
    user_sim = sum(AA,2);
    BB = userSimM * userSimM';
    
    for l2 = 0.0006
        for l1 = 0.006
            L = L + 1;
            
            term = l1*AA + l2*BB;
            pre1 = term*inv(term+I)*train_data;
            pre1 = pre1.*(~train_data);
%             pre1 = train_data*train_data'*pre1;
            %***********************************对测试集数据处理成和训练集为相同规模的矩阵******************************************
            %如果test_data的size大于td矩阵，说明test中有td中没有涉及到的用户（商品），那么这些商品应该去掉
            [m1,n1]=size(pre1);
            [m2,n2]=size(test_data);
            if(max(m1,m2)==m1)  %如果推荐集合中用户多于测试集中的用户，在测试集中补上缺失的用户行（值为0）
            test_data(m2+1:m1,:) = 0;
            else
            test_data(m1+1:m2,:) = [];
            end
            if(max(n1,n2)==n1)  %如果推荐集合中用户推荐的物品多于测试集中的物品，则在测试集中补上缺失的物品（值为0）
            test_data(:,n2+1:n1) = 0;
            else
            test_data(:,n1+1:n2) = [];
            end
            clear  m1 m2 n1 n2;
            %******************************进行指标评价****************************************************************************
           %对推荐结果进行排序(排序条件 测试集矩阵和推荐矩阵规模相同，且除去目标用户浏览过的物品)
           r1 = 0;
           recall=0;
           pre=0;
           plist = zeros(m,3);
           number = 0;
           number1=0;
           aupr = 0;
         
           %遍历训练集中的用户
           for i = 1:m
            if(any(index_==i)==0) %any检查矩阵中是否有非零元素
               % if train_user_du(i)<=threshold
                    temp = (train_data(i,:)==0);
                    number = number+nnz(test_data(i,temp));
                    number1=number1+1;
                    sp = getPrecison(test_data(i,temp),pre1(i,temp),recL);
                    score_list = pre1(i,:);
                    train_list = train_data(i,:);
                    test_list = test_data(i,:);
                    [~, ~,~, aupr1] = calPreRecall(score_list', train_list', test_list');
                    pre=sp+pre;         
                    sre=getRecal(test_data(i,temp),pre1(i,temp),recL);                        
                    recall = recall + sre;
                    aupr = aupr + aupr1;
               % end
            end
           end
          indexResult(L,1)=aupr/number1;
          indexResult(L,2)=pre/number1;
          indexResult(L,3)=recall/number1;
          indexResult(L,4)=2*(recall/number1)*(pre/number1)/(recall/number1+pre/number1); 
          %H I Pop
          [indexResult(L,6),indexResult(L,5),indexResult(L,7)]=getSystemIndex(train_data,train_item_du,pre1,recL,UserExistsBoth,m,n);
          
          indexResult(L,:) %auc, rs, aupr, precision, recall, F, 
          clear pre1;
        end
    end
end
%result ouput
mean(indexResult(1:simulation,:),1)

disp('运行完毕')
