function [pre_list, recall_list,aupr, aupr1] = calPreRecall(score_list, train_list, test_list)
    score_list = score_list + 0.0001; %keep the zero candidate edges
    %Accuracy=TP+TN/(TP+TN+FP+FN)
    %Precision=TP/(TP+FP)
    %输入，scoreMat, disease对应的行似然值；train对应为disease对应的行train,
    %test为disease对应的行test
    
    
    score_list = score_list.*~train_list;%only consider candiate edges
    [row, ~, weight] = find(score_list);
    test_label = test_list(row);
    [~, y] = sort(weight,'descend');
    candidate_len = length(weight);
    tnum = nnz(test_list);
    pre_num = 1:candidate_len;
    pre_list = tnum./pre_num;
    recall_list = ones(1,candidate_len);
    correct_rate = 0;
    for j = 1:candidate_len
        if test_list(row(y(j)))>0
            correct_rate = correct_rate + 1;
        end
        recall_list(j)=correct_rate/tnum;
        pre_list(j)=correct_rate/j;
        if correct_rate==tnum
            break;
        end
    end
    
    roc_y = test_list(row(y));
    %% 求x轴recall的各个点，求y轴precison的各个点。求PR曲线下面积aupr
    P=[1:length(roc_y)]';   %实际上是求(TP+FP)，即所有预测为阳的个数。因为阈值已是降序排序，阈值对应的下标即（TP+FP）
    stack_x = cumsum(roc_y == 1)/sum(roc_y == 1); %x轴：TPR=recall=TP/(TP+FN)=预测为阳的正类/所有正类
    stack_y = cumsum(roc_y == 1)./P; %y轴：precision=TP/(TP+FP)=预测为阳的正类/所有预测为阳
    aupr=sum((stack_x(2:length(roc_y))-stack_x(1:length(roc_y)-1)).*stack_y(2:length(roc_y)));  %PR曲线下面积
    aupr1 = trapz(recall_list, pre_list);
end
    