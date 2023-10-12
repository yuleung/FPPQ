function [W]=WStep(CODEBOOK,CODE,lamada,Y,R)
X_ba=CODEBOOK*CODE;
W=(eye(R)/(X_ba*X_ba'+lamada*eye(R)))*X_ba*Y';
