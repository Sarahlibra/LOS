%%   HD �����û��ĺ�������,�����û����Ƽ��Ĳ�ͬ�̶ȡ����Ƽ��б�Ĳ�ͬ�̶�. ָ�����Ƽ��б�L֮�����Ʒ
%     Instra �ڲ��û��Ƽ��Ĳ�ͬ�̶�
%     pop    �û��Ƽ������г̶�
% ���룺 1.ѵ����  2.Ԥ�⼯���������Ʒ�б� 3,�Ƽ��б�ĳ���  �û����� ��Ʒ����

function  [h Instra Pop allh]=getSystemIndex(traindata,itemdu,pre,L,UserExistBoth,m,n)
    pre=pre(UserExistBoth,:);
    userSize=size(pre,1);

    %% ��������
    %��ȡ�Ƽ��б�ǰL����Ʒ
    for i=1:userSize
        temp=pre(i,:);
        [~,C]=sort(pre(i,:));
        temp(C(1:n-L))=0;
        pre(i,:)=temp;
    end
     pre(pre>0)=1;
     clear temp C;

    %��pre�������û�֮����������
     p=1-(pre*pre'./L);
     p=p-p.*eye(userSize);
     allh = sum(p,1);
     h=sum(sum(p,2),1);
     h=h/(userSize*(userSize-1));
     clear p;
     %% �ڲ����� �������û����Ƽ�����Ʒ֮���������
     Instra=0;
     ItemFenmu=sqrt(itemdu'*itemdu);
     ItemSim=traindata'*traindata./ItemFenmu;
     clear ItemFenmu;
     ItemSim(isinf(ItemSim))=0;
     ItemSim(isnan(ItemSim))=0;
     %ÿ����Ʒ���ж����û���ͬ����
     comm = pre'*pre.*ItemSim;
     Instra1 = (sum(sum(comm,2),1)-trace(comm))/2;
     clear  ItemSim s;
     Instra=Instra1*2/(L*(L-1)*userSize);

     %% �����Ƽ���ӱ�ԣ�����Ʒ��ƽ����
     Pop = sum(pre*(itemdu'),1)/(userSize*L);
     clear pre;
end
 
 