function[f]= objectfunval( XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e)
N=size(XTrain,2);
x_encoding=CODEBOOK*CODE;
lookup=CODEBOOK'*CODEBOOK;
subE=calculate(lookup,CODE,e,N);
f = sum(sum((Y-W'*x_encoding).^2))+lamada*sum(sum(W.^2))+gama*sum(sum((x_encoding-P'*XTrain).^2))+miyou*sum(subE.^2);
end

function[subE] =calculate(lookup,CODE,e,N)
subE=zeros(1,N);
parfor i=1:N
    index=find(CODE(:,i)==1);
subE(1,i)=(2*(lookup(index(1),index(2))+lookup(index(1),index(3))+lookup(index(1),index(4))+lookup(index(2),index(3))+lookup(index(2),index(4))+lookup(index(3),index(4)))-e);
end
end