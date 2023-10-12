function[CODEBOOK]=CStep(CODEBOOK)
options = struct('GradObj','on','Display','off','MaxIter','30','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',1,'GradConstr',false);
tic
[CODEBOOK,~] = fminlbfgs(@objectiveFandG,CODEBOOK,options);
toc
end
