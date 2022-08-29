function [ train test ] = DivideNet(net, ratioTrain, direction)

    %hide the probe set
    
    train = net;
   
    if direction == 0
        [i,j,k]=find(tril(net));
        probe_size=ceil(nnz(net)* (1-ratioTrain)/2);
    
        for pp=1:probe_size
            rand_num=ceil(length(i)*rand(1));
            select_ID1=i(rand_num);
            select_ID2=j(rand_num);
            i(rand_num)=[];
            j(rand_num)=[];
            train(select_ID1,select_ID2)=0;
            train(select_ID2,select_ID1)=0;
        end
    else
        [i,j,k]=find(net);
         probe_size=ceil(nnz(net)* (1-ratioTrain));
    
        for pp=1:probe_size
            rand_num=ceil(length(i)*rand(1));
            select_ID1=i(rand_num);
            select_ID2=j(rand_num);
            i(rand_num)=[];
            j(rand_num)=[];
            train(select_ID1,select_ID2)=0;
        end
    end
    test = net - train; 
   
end
