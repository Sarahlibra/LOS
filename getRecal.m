%%  求单个用户的命中精度,即推荐列表中命中的物品个数
% 输入：1. 单个用户测试集里浏览的物品列表       2.预测集里浏览的物品列表 3,推荐列表的长度

function  recal=getRecal(test,pre,L)
recal=0;
[~,C]=sort(pre);  %B记录值，C记录位置

hitItem=C(1,length(C)-L+1:length(C));
if nnz(test)~=0
    recal=nnz(test(1,hitItem))/nnz(test);
end
clear C hitItem;