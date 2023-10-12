function[e]= eStep(CODEBOOK,CODE,M)
N=size(CODE,2);
lookup=CODEBOOK'*CODEBOOK;
subE=zeros(1,N);
parfor i=1:N
    index=find(CODE(:,i)==1);
    subE(1,i)=2*(lookup(index(1),index(2))+lookup(index(1),index(3))+lookup(index(1),index(4))+lookup(index(2),index(3))+lookup(index(2),index(4))+lookup(index(3),index(4)));
end
e=sum(subE')/N;