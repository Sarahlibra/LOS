%%  �󵥸��û������о���,���Ƽ��б������е���Ʒ����
% ���룺1. �����û����Լ����������Ʒ�б�       2.Ԥ�⼯���������Ʒ�б� 3,�Ƽ��б�ĳ���

function  recal=getRecal(test,pre,L)
recal=0;
[~,C]=sort(pre);  %B��¼ֵ��C��¼λ��

hitItem=C(1,length(C)-L+1:length(C));
if nnz(test)~=0
    recal=nnz(test(1,hitItem))/nnz(test);
end
clear C hitItem;