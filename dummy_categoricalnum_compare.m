%% This code loads the data, and uses two approches to test the data set
% Approach 1 - Converts all the string values to numeric values and applies
% Decesion tree on that data
% Approach 2 - Converts all the string values to dummy values and applies
% Decesion tree on that data
% At last there is comparision of both the data set


clc; clear all; 

%% Load Data

data = ImportBankData('bank-full.csv');
Var = data.Properties.VarNames;

%% We are applying here Label Encoding to convert all the string values 
% to numeric values

data = datasetfun(@removequotes,data,'DatasetOutput',true);

[row,col] = size(data);
cat = false(1,col);
for a = 1:col
    if isa(data.(Var{a}),'cell') || isa(data.(Var{a}),'nominal')
        cat(a) = true;
        data.(Var{a}) = nominal(data.(Var{a}));
    end
end

Pred = cat(1:end-1);
rng('default');
disp(Var(cat));
%% Divide data in Training and Testing Set

Y = data.y;
X = double(data(:,1:end-1));
disp('Marketing Campaign')
tabulate(Y)

cvdiv = cvpartition(length(data),'holdout',0.40);

Xtrain = X(training(cvdiv),:);
Ytrain = Y(training(cvdiv),:);
Ytrain = nominal(Ytrain);
Xtest = X(test(cvdiv),:);
Ytest = Y(test(cvdiv),:);
Ytest = nominal(Ytest);

disp('Training Set')
tabulate(Ytrain)
disp('Test Set')
tabulate(Ytest)

%% Test the Model and Evaluate Performance by Decision Tree
% with Label Encoding

Modl_decT = fitctree(Xtrain,Ytrain,'CategoricalPredictors',Pred);
[Y_decT, scores] = Modl_decT.predict(Xtest);
CM_decT = confusionmat(Ytest,Y_decT);
CM_decT = bsxfun(@rdivide,CM_decT,sum(CM_decT,2)) * 100;

%% Load Data

bank = ImportBankData('bank-full.csv');
Var_d = bank.Properties.VarNames;

%% We are applying here Dummy Encoding to convert all the string values 
% to Dummy Values

bank.job = categorical(bank.job);
bank.marital = categorical (bank.marital);
bank.education = categorical (bank. education);
bank.default=categorical(bank.default);
bank.housing=categorical(bank.housing);
bank.loan=categorical(bank.loan);
bank.contact=categorical(bank.contact);
bank.month=categorical(bank.month);
bank.poutcome=categorical(bank.poutcome);
    
bank.job = dummyvar(bank.job);
bank.marital = dummyvar (bank.marital);
bank.education = dummyvar (bank. education);
bank.default=dummyvar(bank.default);
bank.housing=dummyvar(bank.housing);
bank.loan=dummyvar(bank.loan);
bank.contact=dummyvar(bank.contact);
bank.month=dummyvar(bank.month);
bank.poutcome=dummyvar(bank.poutcome);

%% Divide data in Training and Testing Set

Y_d = bank.y;
X_d = double(bank(:,1:end-1));
disp('Marketing Campaign')
tabulate(Y_d)

cvdiv_d = cvpartition(length(bank),'holdout',0.40);

Xtrain_d = X_d(training(cvdiv_d),:);
Ytrain_d = Y_d(training(cvdiv_d),:);
Ytrain_d = nominal(Ytrain_d);
Xtest_d = X_d(test(cvdiv_d),:);
Ytest_d = Y_d(test(cvdiv_d),:);
Ytest_d = nominal(Ytest_d);

disp('Training Set')
tabulate(Ytrain_d)
disp('Test Set')
tabulate(Ytest_d)

%% Test the Model and Evaluate Performance by Decision Tree
% with Dummy Encoding

Modl_decT_d = fitctree(Xtrain_d,Ytrain_d,'CategoricalPredictors',51);
[Y_decT_d,scores_d] = Modl_decT_d.predict(Xtest_d);
CM_decT_d = confusionmat(Ytest_d,Y_decT_d);
CM_decT_d = bsxfun(@rdivide,CM_decT_d,sum(CM_decT_d,2)) * 100;

%% Compare the performance of Decision Tree with 
% Label Encoding and Dummy Encoding

C_DT = [CM_decT CM_decT_d];
Model_Output = {'Decision Tree with Label Encoding','Decision Tree with Dummy Variables '};
comparisonPlot( C_DT, Model_Output )
