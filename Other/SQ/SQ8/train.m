clc;clear;

load('db.mat');
XTrain = db_feature';
YLabel = db_label;

%并行计算开启
distcomp.feature( 'LocalUseMpiexec', false );

%data init
disp(size(XTrain))
disp(size(YLabel))

lamada=1;gama=0.01;miyou=0.1;

#compress
R=256;

%Seg Num
M=8;

%num of center each Seg
K=256;

%iterations
NITS=2;

%num of class
C=1000;

%constant inter-dictionaryelement-product
e=0;

N=size(XTrain,2);Y=zeros(C,N);W=randn(R,C);
for i=1:N
    Y(YLabel(1,i)+1,i)=1;
end

%P通过PCA初始化, CODEBOOK和CODE通过PQ方法初始化
[P,CODE,CODEBOOK]=initialize(XTrain,R,K,M);
% load('initial.mat');
disp('initial compl')
for i=1:NITS
    disp(['迭代次数：',num2str(i),]);
    qerror(i,1)=objectfunval( XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e);
    W=WStep(CODEBOOK,CODE,lamada,Y,R);
    disp('WStep Done!')
    qerror(i,2)=objectfunval(XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e);
    P= PStep(CODEBOOK,CODE,XTrain);
    disp('PStep Done!')
    qerror(i,3)=objectfunval( XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e);
    e= eStep(CODEBOOK,CODE,M);
    disp('eStep Done!')
    qerror(i,4)=objectfunval( XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e);
    %通过mat文件把参数传递给objectiveF.m函数

    disp(size(XTrain))
    disp(size(CODE))
    disp(size(Y))
    disp(size(CODEBOOK))
    save('parameterToCStep.mat', 'W','P','e','miyou','gama','R','CODE','K','lamada');

    %'CODE','XTrain','Y'
    %CODE1 = CODE(1:256, :);
    %CODE2 = CODE(257:512,:);
    %CODE3 = CODE(513:768,:);
    %CODE4 = CODE(769:1024, :);


    %save('CODE1.mat', 'CODE1');
    %save('CODE2.mat', 'CODE2');
    %save('CODE3.mat', 'CODE3');
    %save('CODE4.mat', 'CODE4');

    Y1 = Y(1:250, :);
    Y2 = Y(251:500,:);
    Y3 = Y(501:750,:);
    Y4 = Y(751:1000, :);

    save('Y1.mat', 'Y1');
    save('Y2.mat', 'Y2');
    save('Y3.mat', 'Y3');
    save('Y4.mat', 'Y4');

    CODEBOOK= CStep(CODEBOOK);

    disp('CStep Done!');
    delete('parameterToCStep.mat');
    %delete('CODE1.mat');
    %delete('CODE2.mat');
    %delete('CODE3.mat');
    %delete('CODE4.mat');
    delete('Y1.mat');
    delete('Y2.mat');
    delete('Y3.mat');
    delete('Y4.mat');


    qerror(i,5)=objectfunval( XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e);
    CODE=BStep(Y,W,CODEBOOK,CODE,P,XTrain,e,miyou,gama,M,K);
    disp('BStep Done!')
    qerror(i,6)=objectfunval( XTrain, CODE, CODEBOOK ,P,Y,W,gama,miyou,lamada,e);
    disp([num2str(qerror(i,1)),'  ',num2str(qerror(i,2)),'  ',num2str(qerror(i,3)),'  ',num2str(qerror(i,4)),'  ',num2str(qerror(i,5)),'  ',num2str(qerror(i,6))]);
end


%Test
load('test.mat');
testdata = test_feature';
testgnd = test_label+1;

NUM_TEST=size(testdata,2)
test_tran=P'*testdata;

%construct The Table of look-up
for i=1:NUM_TEST
    aa=repmat(test_tran(:,i),[1,K*M]);
    lookup_table(:,i)=sum((aa-CODEBOOK).^2)';
end

result=lookup_table'*CODE;

% hamming ranking: MAP
[~, distanceRank]=sort(result,2);

[train_gnd,~]=find(Y==1);

MAP = cat_apcal(train_gnd,testgnd,distanceRank');
disp(['MAP is ',num2str(MAP)]);
