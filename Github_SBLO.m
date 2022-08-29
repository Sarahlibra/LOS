%% ʵ��HHP�㷨
clear;clc;
%% �����û�CH2-L2���ƶȾ���
userSim = load('Data\\friendfeed\\social_data.txt');
% userSim(:,3) = userSim(:,3)./sum(userSim(:,3));
userSimM = full(spconvert(userSim));
%��ȫ����
[m,n]=size(userSimM);
if m > n
    userSimM(:,n+1:m) = 0;
elseif m < n
    userSimM(m+1:n,:)=0;
end
userSimM = userSimM + userSimM';
userSimM(userSimM~=0)=1;
%% �������ݣ�ת���ɾ���, ����ѵ���������Լ�
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

    %��������û�  ��Ʒ ��� ת��Ϊ�û�*��Ʒ��(0,1)��ά����
    train_data=full(train);  %ѵ�����û�����Դ�������
    test_data=full(test);
    clear train test;
    train_user_du=sum(train_data,2); %��ȡ�û���
    test_user_du=sum(test_data,2);
    train_item_du = sum(train_data,1);%��Ʒ�Ķ�
    [m,n]=size(train_data);
    %��ȡ���Լ�����ѵ����û�д�ֵ��û�
    index1 = find(train_user_du==0);
    index2 = find(test_user_du==0);
    index_ = unique(union(index1,index2)); 
    clear index1 index2;
    index4 = find(test_user_du~=0);
%     UserExistsBoth = intersect(index3,index4);%%%��������ָ��ļ���
    UserExistsBoth = index4;
    clear index4;

    %��ȡ��һ������ǰTopN���ھ��û�
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
            %***********************************�Բ��Լ����ݴ���ɺ�ѵ����Ϊ��ͬ��ģ�ľ���******************************************
            %���test_data��size����td����˵��test����td��û���漰�����û�����Ʒ������ô��Щ��ƷӦ��ȥ��
            [m1,n1]=size(pre1);
            [m2,n2]=size(test_data);
            if(max(m1,m2)==m1)  %����Ƽ��������û����ڲ��Լ��е��û����ڲ��Լ��в���ȱʧ���û��У�ֵΪ0��
            test_data(m2+1:m1,:) = 0;
            else
            test_data(m1+1:m2,:) = [];
            end
            if(max(n1,n2)==n1)  %����Ƽ��������û��Ƽ�����Ʒ���ڲ��Լ��е���Ʒ�����ڲ��Լ��в���ȱʧ����Ʒ��ֵΪ0��
            test_data(:,n2+1:n1) = 0;
            else
            test_data(:,n1+1:n2) = [];
            end
            clear  m1 m2 n1 n2;
            %******************************����ָ������****************************************************************************
           %���Ƽ������������(�������� ���Լ�������Ƽ������ģ��ͬ���ҳ�ȥĿ���û����������Ʒ)
           r1 = 0;
           recall=0;
           pre=0;
           plist = zeros(m,3);
           number = 0;
           number1=0;
           aupr = 0;
         
           %����ѵ�����е��û�
           for i = 1:m
            if(any(index_==i)==0) %any���������Ƿ��з���Ԫ��
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

disp('�������')