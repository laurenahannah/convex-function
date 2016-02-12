% Learn to use CAP!

% We are going to test on Problem 2

p = [0.0680, 0.0160, 0.1707, 0.1513, 0.1790, 0.2097, 0.0548, 0.0337, 0.0377, 0.0791];

nIID = 10000;

num = [100, 200, 500, 1000, 2000, 5000, 10000];
Mvec = [100];

numTrials = 10;

mse = zeros(numTrials,2,length(num));
mae = zeros(numTrials,2,length(num));
ti = zeros(numTrials,2,length(num));
% [CAP, fastCAP]

yHatOut = [];

xSaveTrain = [];
ySaveTrain = [];
ySaveTest = [];
xSaveTest = [];

counter = 1;

for m = 1:numTrials
    disp('=======================')
    disp('m = ')
    disp(m)
    disp('=======================')
    
    xTrain = randn(max(num),10);
    yTrain = exp(xTrain*p') + 0.1*randn(max(num),1);
    xTest = randn(10000,10);
    yTest = exp(xTest*p');
    xSaveTest = [xSaveTest; xTest];
    ySaveTest = [ySaveTest; yTest];
    xSaveTrain = [xSaveTrain; xTrain];
    ySaveTrain = [ySaveTrain; yTrain];

for i = 1:length(num)
    n = num(i)
    % Do regular CAP
    if n < 10^6
        tstart = tic;
        [alpha, beta, K] = CAP(xTrain(1:n,:),yTrain(1:n));
        tend = toc(tstart);
        disp('CAP done')
        yHat = convexEval(alpha,beta,xTest);
        res = yTest(1:nIID) - yHat(1:nIID);
        ti(m,counter,i) = tend;
        mse(m,counter,i) = mean(res.^2);
        mae(m,counter,i) = mean(abs(res));
    end
        counter = counter+1;
    % Do Fast CAP (new)
        tstart = tic;
        [alpha, beta, K] = CAPfast(xTrain(1:n,:),yTrain(1:n));
        tend = toc(tstart);
        disp('Fast CAP done')
        yHat = convexEval(alpha,beta,xTest);
        res = yTest(1:nIID) - yHat(1:nIID);
        figure
        plot(yTest,yHat,'.')
        ti(m,counter,i) = tend;
        mse(m,counter,i) = mean(res.^2);
        mae(m,counter,i) = mean(abs(res));
        counter = counter+1; 
    
    
    counter = 1;
    
end
maeHat = mae(m,:,:);
mmae = reshape(maeHat,2,length(num));
mseHat = mse(m,:,:);
mmse = reshape(mseHat,2,length(num));

tiHat = ti(m,:,:);
mti = reshape(tiHat,2,length(num));

mmae
mmse
mti

end

mmean = mean(mse,1);
mmmean = reshape(mmean,2,length(num))
mstd = std(mse,1);
mmstd = reshape(mstd,2,length(num))

mtime = mean(ti,1);
mmtime = reshape(mtime,2,length(num))


    