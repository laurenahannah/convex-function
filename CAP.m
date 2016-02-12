function [alpha, beta, abStruct] = CAP(x,y,convexFlag,logMult,knotNum)

% CAPnew runs the CAP algorithm on input data x,y to create a piecewise
% linear convex or concave function
%
% Inputs:
%   - x             : training data covariates (n by d)
%   - y             : training data response (n by 1)
%   - convexFlag    : 0 if convex, 1 if concave (convex by default)
%   - logMult       : multiplier of log(n) decides minimal number of
%                     observations per partition; default is 3
%
% Outputs:
%   - alpha         : vector of intercepts for CAP estimator
%   - beta          : vector of slopes for estimator
%   - abStruct      : structure array of (alpha,beta), residuals,
%                     partitions and BIC for k = 1,...,K; save using "Save
%                     Workspace"


[n,d] = size(x);

if ((nargin == 2)||(isempty(convexFlag)))
    convexFlag = 0;
end
if ((nargin < 4)||(isempty(logMult)))
    logMult = 5;
end
if ((nargin < 5)||(isempty(knotNum)))
    knotNum = 10;
end

minNode = max(n/(logMult*log(n)),2*(d+1));

alpha = [];
beta = [];
abStruct = [];

keepGoing = true;

% Initialize with one partition
K = 1;
if (n > 2*(d+1))
    ab = regress(y,[ones(n,1), x]);
else
    ab = zeros(d+1);
    ab(1) = mean(y);
end

g = [ones(n,1), x]*ab;
res = y - g;
varG = getVar(y - g,K,d);
    
BIC = mean((y-g).^2)/(1-K*(d+1)/n)^2;

alpha = ab(1);
beta = ab(2:end);
idList = ones(n,1); % Parition of observations

abStruct(1).alpha = alpha;
abStruct(1).beta = beta;
abStruct(1).residuals = res;
abStruct(1).idList = idList;

BICvec = [];

BICvec = [BICvec; BIC];
MSEvec = [];
MSEvec = [MSEvec; mean(res.^2)];
GCVvec = [];
GCVvec = [GCVvec; sum(res.^2)/(n*(1-(K*(d+1))/n)^2)];

abStruct(1).BIC = BIC;

if (n < max(2*n/(logMult*log(n)), 4*(d+1)))
    keepGoing = false;
end

resStruct(1).res = res; % Structure array for residuals for each subset

while(keepGoing)
    %K
    % Loop through subsets
    counter = 1;
    idListOld = abStruct(K).idList;
    tempStruct = []; % For storing proposal parameters
    resVec = []; % For storing and comparing residuals
    for k = 1:K
        idVec = find(idListOld == k);
        nk = length(idVec);
        xHat = x(idVec,:);
        yHat = y(idVec);
        alphaH = alpha;
        betaH = beta;
        alphaH(k) = [];
        betaH(:,k) = [];
        if (nk >= max(2*n/(logMult*log(n)), 4*(d+1)))
            % Loop through dimensions
            for j = 1:d
                % Create grid for searching
                minVec = min(xHat(:,j));
                maxVec = max(xHat(:,j));
                if (maxVec - minVec < 0.0001) % We need to jitter
                    xHat(:,j) = xHat(:,j) + 0.0001*randn(nk,1);
                    minVec = min(xHat(:,j));
                    maxVec = max(xHat(:,j));
                end
                knotStep = (maxVec - minVec)/knotNum;
                knotMat = zeros(1,knotNum+1);
                knotMat = minVec:knotStep:maxVec;
                for ell = 2:knotNum
                    % Make sure that both parts of the partition have
                    % enough data
                    xHat1Ind = find(xHat(:,j) <= knotMat(ell));
                    xHat2Ind = find(xHat(:,j) > knotMat(ell));
                    n1 = length(xHat1Ind);
                    n2 = length(xHat2Ind);
                    if ((n1 > minNode)&&(n2 > minNode))
                        % Then we can compute betas
                        ab1 = regress(yHat(xHat1Ind),[ones(n1,1),xHat(xHat1Ind,:)]);
                        ab2 = regress(yHat(xHat2Ind),[ones(n2,1), xHat(xHat2Ind,:)]);
                        betaHat = [betaH, ab1(2:end), ab2(2:end)];
                        alphaHat = [alphaH, ab1(1), ab2(1)];
                        if convexFlag == 0
                            [gg, iList] = max([ones(n,1),x]*[alphaHat; betaHat],[],2);
                        else
                            [gg, iList] = min([ones(n,1),x]*[alphaHat; betaHat],[],2);
                        end
                        resNew1 = yHat(xHat1Ind) - [ones(n1,1), xHat(xHat1Ind,:)]*ab1;
                        resNew2 = yHat(xHat2Ind) - [ones(n2,1), xHat(xHat2Ind,:)]*ab2;
                        resDiff = mean((y - gg).^2);
                        tempStruct(counter).k = k;
                        tempStruct(counter).ab1 = ab1;
                        tempStruct(counter).ab2 = ab2;
                        tempStruct(counter).res1 = resNew1;
                        tempStruct(counter).res2 = resNew2;
                        idListNew = idListOld;
                        idHi = find(x(:,j) > knotMat(ell));
                        idChange = intersect(idHi,idVec);
                        idListNew(idChange) = (K+1)*ones(length(idChange),1); % Move hi residuals into subset K+1
                        tempStruct(counter).idList = idListNew;
                        resVec = [resVec; resDiff];
                        counter = counter + 1;
                    end
                end
                if (counter == 1) % We need to split by the median
                    xMed = median(xHat(:,j));
                    xHat1Ind = find(xHat(:,j) <= xMed);
                    xHat2Ind = find(xHat(:,j) > xMed);
                    n1 = length(xHat1Ind);
                    n2 = length(xHat2Ind);
                    if ((n1 > 0)&&(n2 > 0))
                        % Then we can compute betas
                        ab1 = regress(yHat(xHat1Ind),[ones(n1,1),xHat(xHat1Ind,:)]);
                        ab2 = regress(yHat(xHat2Ind),[ones(n2,1), xHat(xHat2Ind,:)]);
                        betaHat = [betaH, ab1(2:end), ab2(2:end)];
                        alphaHat = [alphaH, ab1(1), ab2(1)];
                        if convexFlag == 0
                            [gg, iList] = max([ones(n,1),x]*[alphaHat; betaHat],[],2);
                        else
                            [gg, iList] = min([ones(n,1),x]*[alphaHat; betaHat],[],2);
                        end
                        resNew1 = yHat(xHat1Ind) - [ones(n1,1), xHat(xHat1Ind,:)]*ab1;
                        resNew2 = yHat(xHat2Ind) - [ones(n2,1), xHat(xHat2Ind,:)]*ab2;
                        resDiff = mean((y - gg).^2);
                        tempStruct(counter).k = k;
                        tempStruct(counter).ab1 = ab1;
                        tempStruct(counter).ab2 = ab2;
                        tempStruct(counter).res1 = resNew1;
                        tempStruct(counter).res2 = resNew2;
                        idListNew = idListOld;
                        idHi = find(x(:,j) > xMed);
                        idChange = intersect(idHi,idVec);
                        idListNew(idChange) = (K+1)*ones(length(idChange),1); % Move hi residuals into subset K+1
                        tempStruct(counter).idList = idListNew;
                        resVec = [resVec; resDiff];
                        counter = counter + 1;
                    end
                end
            end
        end
    end
    
    % Pick the split that maximizes the change in the (local) residuals
    [val, choice] = min(resVec);
    
    % Record stuff for that choice
    alpha = abStruct(K).alpha;
    beta = abStruct(K).beta;
    res = abStruct(K).residuals;
    k = tempStruct(choice).k;
    ab1 = tempStruct(choice).ab1;
    ab2 = tempStruct(choice).ab2;
    res1 = tempStruct(choice).res1;
    res2 = tempStruct(choice).res2;
    idList = tempStruct(choice).idList;
    alpha(k) = ab1(1);
    beta(:,k) = ab1(2:end);
    alpha = [alpha, ab2(1)];
    beta = [beta, ab2(2:end)];
    indsChoice1 = find(idList == k);
    indsChoice2 = find(idList == K+1);
    res(indsChoice1) = res1;
    res(indsChoice2) = res2;
    
    K = K + 1;
    
    abStruct(K).alpha = alpha;
    abStruct(K).beta = beta;
    abStruct(K).idList = idList;
    abStruct(K).residuals = res;
    
    % Refit
    refit = true;
    if (convexFlag == 0)
        [g, iiList] = max([ones(n,1), x]*[alpha; beta],[],2);
    else
        [g, iiList] = min([ones(n,1), x]*[alpha; beta],[],2);
    end
    nVec = zeros(K,1);
    k = 1;
    alphaProp = [];
    betaProp = [];
    res = zeros(n,1);
    nextItem = true;
    while ((refit)&&(nextItem))
        ii = find(iiList == k);
        if (length(ii) < max(n/(logMult*log(n)), 2*(d+1)))
            refit = false;
        else
            ab = regress(y(ii),[ones(length(ii),1), x(ii,:)]);
            alphaProp = [alphaProp, ab(1)];
            betaProp = [betaProp, ab(2:end)];
            res(ii) = y(ii) - [ones(length(ii),1), x(ii,:)]*ab;
        end
        if (k == K)
            nextItem = false;
        else
            k = k + 1;
        end
    end
         
    if (refit)
        idList = iiList;
        abStruct(K).alpha = alphaProp;
        abStruct(K).beta = betaProp;
        abStruct(K).idList = idList;
        abStruct(K).residuals = res;
    end
    
    % Get BIC and determine stopping
    if (convexFlag == 0)
        g = max([ones(n,1), x]*[alpha; beta],[],2);
    else
        g = min([ones(n,1), x]*[alpha; beta],[],2);
    end
    
    nVec = zeros(K,1);
    
    kk = K*(d+1);
    varG = getVar(y - g,K,d);
    
    %BIC = log(n)*K*(d+1)/n*varG;
    BIC = 0;
    
    resStruct = [];
    res = abStruct(K).residuals;
    
    
    BIC = 0;
    
    resStruct = [];
    
    
    for k = 1:K
        ii = find(idList == k);
        nVec(k) = length(ii);
        resStd = std(g(ii));
        %BIC = BIC + sum((y(ii)-g(ii)).^2)/(1-(d+1)/nVec(k))^2;
        alphaHat = alpha;
        betaHat = beta;
        alphaHat(k) = alphaHat(k)/(1-(d+1)/nVec(k));
        betaHat(:,k) = betaHat(:,k)/(1-(d+1)/nVec(k));
        if (convexFlag == 0)
            [ggg, iHat] = max([ones(nVec(k),1), x(ii,:)]*[alphaHat; betaHat],[],2);
        else
            [ggg, iHat] = min([ones(nVec(k),1), x(ii,:)]*[alphaHat; betaHat],[],2);
        end
        divVec = (iHat == k);
        BIC = BIC + sum(((y(ii)-ggg).^2)./((ones(nVec(k),1) - (d+1)/nVec(k)*divVec).^2));
    end
    abStruct(K).BIC = BIC;
    BICvec = [BICvec; BIC/n];
    MSEvec = [MSEvec; mean((y-g).^2)];
    GCVvec = [GCVvec; sum((y-g).^2)/(n*(1-(K*(d+1))/n)^2)];
    % See if we should stop
    if (max(nVec) < max(2*n/(logMult*log(n)), 4*(d+1)))
        keepGoing = false;
    end
end

% Pick the value that minimizes the BIC

[val, choice] = min(BICvec);
alpha = abStruct(choice).alpha;
beta = abStruct(choice).beta;
%disp('GCV:')
%disp(GCVvec')
%disp('BIC:')
%disp(BICvec')
%disp('MSE:')
%disp(MSEvec')
    
function sigma2 = getVar(y,k,d)

n = length(y);
yBar = mean(y);
sigma2 = n/(n-k*(d+1))*mean((y-yBar).^2);