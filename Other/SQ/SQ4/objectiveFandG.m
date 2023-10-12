function [f,g] = objectiveFandG(CODEBOOK)
load('parameterToCStep.mat');

%load('train_1.mat');
%load('train_2.mat');
%load('train_3.mat');
%XTrain = [train_feature_1; train_feature_2; train_feature_3]';
load('db.mat');
XTrain = db_feature';


load('Y1.mat');
load('Y2.mat');
load('Y3.mat');
load('Y4.mat');
Y = [Y1; Y2; Y3; Y4];


%load('CODE1.mat');
%load('CODE2.mat');
%load('CODE3.mat');
%load('CODE4.mat');
%CODE = [CODE1; CODE2; CODE3; CODE4];


N=size(XTrain,2);
lookup=CODEBOOK'*CODEBOOK;
x_encoding=CODEBOOK*CODE;
subE=calculate(lookup,CODE,e,N);

f = sum(sum((Y-W'*x_encoding).^2))+lamada*sum(sum(W.^2))+gama*sum(sum((x_encoding-P'*XTrain).^2))+miyou*sum(subE.^2);
if nargout > 1
    bn1=CODE(1:256,:);
    bn2=CODE(257:512,:);
    bn3=CODE(513:768,:);
    bn4=CODE(769:1024,:);

    index=zeros(4,N);

    for i=1:N
        %disp(i)
        %disp(size(find(CODE(:,i)==1)))
        index(:,i)=find(CODE(:,i)==1);

    end

    temp21=x_encoding-CODEBOOK(:,index(1,:));
    copy=repmat(subE,[size(temp21,1),1]);

      g(1:256,1:256)=2*W*(W'*x_encoding-Y)*bn1'+2*gama*(x_encoding-P'*XTrain)*bn1'+4*miyou*(copy.*temp21)*bn1';
    %'
    temp21=x_encoding-CODEBOOK(:,index(2,:));
     copy=repmat(subE,[size(temp21,1),1]);
     g(1:256,257:512)=2*W*(W'*x_encoding-Y)*bn2'+2*gama*(x_encoding-P'*XTrain)*bn2'+4*miyou*(copy.*temp21)*bn2';


    %'
    temp21=x_encoding-CODEBOOK(:,index(3,:));
    copy=repmat(subE,[size(temp21,1),1]);
        %%%%SQ  ʽ ӣ 10    C3 ĵ
     g(1:256,513:768)=2*W*(W'*x_encoding-Y)*bn3'+2*gama*(x_encoding-P'*XTrain)*bn3'+4*miyou*(copy.*temp21)*bn3';


    %'
    temp21=x_encoding-CODEBOOK(:,index(4,:));
    copy=repmat(subE,[size(temp21,1),1]);

    g(1:256,769:1024)=2*W*(W'*x_encoding-Y)*bn4'+2*gama*(x_encoding-P'*XTrain)*bn4'+4*miyou*(copy.*temp21)*bn4';



end
end

function[subE] =calculate(lookup,CODE,e,N)
subE=zeros(1,N);
for i=1:N
    index=find(CODE(:,i)==1);
    subE(1,i)=(2*(lookup(index(1),index(2))+lookup(index(1),index(3))+lookup(index(1),index(4))+lookup(index(2),index(3))+lookup(index(2),index(4))+lookup(index(3),index(4)))-e);
end
