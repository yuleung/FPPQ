function[e]= eStep(CODEBOOK,CODE,M)
N=size(CODE,2);
lookup=CODEBOOK'*CODEBOOK;
subE=zeros(1,N);
for i=1:N
    index=find(CODE(:,i)==1);
    subE(1,i)=2*(lookup(index(1),index(2))+lookup(index(1),index(3))+lookup(index(1),index(4))+lookup(index(1),index(5))+lookup(index(1),index(6))+lookup(index(1),index(7))+lookup(index(1),index(8))+lookup(index(2),index(3))+lookup(index(2),index(4))+lookup(index(2),index(5))+lookup(index(2),index(6))+lookup(index(2),index(7))+lookup(index(2),index(8))+lookup(index(3),index(4))+lookup(index(3),index(5)) + lookup(index(3),index(6))+lookup(index(3),index(7))+lookup(index(3),index(8))+lookup(index(4),index(5)) + lookup(index(4),index(6))+lookup(index(4),index(7))+lookup(index(4),index(8))+lookup(index(5),index(6))+lookup(index(5),index(7))+lookup(index(5),index(8))+lookup(index(6),index(7))+lookup(index(6),index(8))+lookup(index(7),index(8)));
e=sum(subE')/N;
end