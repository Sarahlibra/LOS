%%  �󵥸��û������о���,���Ƽ��б������е���Ʒ����
% ���룺1. �����û����Լ����������Ʒ�б�       2.Ԥ�⼯���������Ʒ�б� 3,�Ƽ��б�ĳ���

function  precison=getPrecison(test,pre,L)

[~,C]=sort(pre);  %B��¼ֵ��C��¼λ��
hitItem=C(1,length(C)-L+1:length(C));
precison=nnz(test(1,hitItem))/L;
clear C hitItem;