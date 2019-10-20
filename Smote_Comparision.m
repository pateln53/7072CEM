%% This code loads the data, converts string value to numeric value,
% divides data in training and testing set. 
% Post that it applies methods (Logistic Regression, Decision Tree, 
% Ensemble (TreeBagger) on that data.
% Then it solves the problem of class imbalance in data so we are applying 
% SMOTE ( Synthetic Minority Over-sampling Technique) 
% algorithm to solve this problem and training all the models again.
% Al last there is comparision of all the models with and without SMOTE to
% see the difference

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

%% Test the Model and Evaluate Performance by Linear Model (Logistic Regression)

Modl_ldm = fitglm(Xtrain,double(Ytrain)-1,'linear','link','logit','Distribution','binomial');
Y_ldm = Modl_ldm.predict(Xtest);
Y_ldm = round(Y_ldm) + 1;
CM_ldm = confusionmat(double(Ytest),Y_ldm);
CM_ldm = bsxfun(@rdivide,CM_ldm,sum(CM_ldm,2)) * 100;

%% Test the Model and Evaluate Performance by Decision Tree

Modl_decT = fitctree(Xtrain,Ytrain,'CategoricalPredictors',Pred);
Y_decT = Modl_decT.predict(Xtest);
CM_decT = confusionmat(Ytest,Y_decT);
CM_decT = bsxfun(@rdivide,CM_decT,sum(CM_decT,2)) * 100;

%% Test the Model and Evaluate Performance by Ensemble Method by Tree Bagger

Mcost = [0 1;5 0];
PC = statset('UseParallel',true);
Modl_tb = TreeBagger(150,Xtrain,Ytrain,'method','classification','categorical',Pred,'Options',PC,'OOBVarImp','on','cost',Mcost);
[Y_tb, classifScore] = Modl_tb.predict(Xtest);
Y_tb = nominal(Y_tb);
CM_tb = confusionmat(Ytest,Y_tb);
CM_tb = bsxfun(@rdivide,CM_tb,sum(CM_tb,2)) * 100;

%% Compare the output of all the above models

C_CM = [CM_ldm CM_decT CM_tb];
Model_Output = {'Logistic Regression ','Decision Trees ', 'TreeBagger '};
comparisonPlot( C_CM, Model_Output )

%% Now the issue is there is misclassification in data so we are applying 
% SMOTE ( Synthetic Minority Over-sampling Technique) 
% algorithm to solve this problem 

Y = double (Y)-1;
[final_features, final_mark] =  SMOTE(X,Y);
disp('SMOTE')
tabulate(final_mark)

%%  Now we have already solved the problem of class imbalance and so
% we will train the data again with all the above mentioned methods 

%% Divide data in Training and Testing Set

bank_S = [final_features final_mark];
cvdiv_S = cvpartition(length(bank_S),'holdout',0.40);

Xtrain_S = final_features(training(cvdiv_S),:);
Ytrain_S = final_mark(training(cvdiv_S),:);
Ytrain_S = nominal(Ytrain_S);
Xtest_S = final_features(test(cvdiv_S),:);
Ytest_S = final_mark(test(cvdiv_S),:);
Ytest_S = nominal(Ytest_S);

disp('Training Set')
tabulate(Ytrain_S)
disp('Test Set')
tabulate(Ytest_S)

%% Test the Model and Evaluate Performance by Linear Model 
% (Logistic Regression) with SMOTE

Modl_ldm_S = fitglm(Xtrain_S,double(Ytrain_S)-1,'linear','link','logit','Distribution','binomial');
Y_ldm_S = Modl_ldm_S.predict(Xtest_S);
Y_ldm_S = round(Y_ldm_S) + 1;
CM_ldm_S = confusionmat(double(Ytest_S),Y_ldm_S);
CM_ldm_S = bsxfun(@rdivide,CM_ldm_S,sum(CM_ldm_S,2)) * 100;

%% Test the Model and Evaluate Performance by Decision Tree with SMOTE

Modl_decT_S = fitctree(Xtrain_S,Ytrain_S,'CategoricalPredictors',Pred);
Y_decT_S = Modl_decT_S.predict(Xtest_S);
CM_decT_S = confusionmat(Ytest_S,Y_decT_S);
CM_decT_S = bsxfun(@rdivide,CM_decT_S,sum(CM_decT_S,2)) * 100;

%% Test the Model and Evaluate Performance by Ensemble Method by Tree Bagger

Mcost_S = [0 1;5 0];
PC_1 = statset('UseParallel',true);
Modl_tb_S = TreeBagger(150,Xtrain_S,Ytrain_S,'method','classification','categorical',Pred,'Options',PC_1,'OOBVarImp','on','cost',Mcost_S);
[Y_tb_S, classifScore_S] = Modl_tb_S.predict(Xtest_S);
Y_tb_S = nominal(Y_tb_S);
CM_tb_S = confusionmat(Ytest_S,Y_tb_S);
CM_tb_S = bsxfun(@rdivide,CM_tb_S,sum(CM_tb_S,2)) * 100;

%% Compare Performance of Logistic Regression With and Without SMOTE

C_LR = [CM_ldm CM_ldm_S];
Model_Output = {'Logistic Regression ','Logistic Regression with SMOTE '};
comparisonPlot( C_LR, Model_Output )

%% Compare Performance of Decision Tree With and Without SMOTE

C_DT = [CM_decT CM_decT_S];
Model_Output = {'Decision Trees ','Decision Trees with SMOTE '};
comparisonPlot( C_DT, Model_Output )


%% Compare Performance of all the Models (Logistic Regression, Decision Tree, 
% Ensemble (TreeBagger)) With and Without SMOTE

C_CM = [CM_ldm CM_ldm_S CM_decT CM_decT_S CM_tb CM_tb_S];
Model_Output = {'Logistic Regression ','Logistic Regression with SMOTE ','Decision Trees ','Decision Trees with SMOTE ', 'TreeBagger ', 'TreeBagger with SMOTE '};
comparisonPlot( C_CM, Model_Output )
