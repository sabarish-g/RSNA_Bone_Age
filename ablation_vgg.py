import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


#################Load the train data########################

train_features = np.load('/home/sxg8458/RSNA_Bone_Age/train_images_4k.npy')
print('The shape of the train features is %s ' %str(train_features.shape))


test_features = np.load('/home/sxg8458/RSNA_Bone_Age/test_images_4k.npy')
print('The shape of the test features is %s ' %str(test_features.shape))


train_labels = np.load('/home/sxg8458/RSNA_Bone_Age/train_labels.npy')
print('The shape of the train labels is %s '%str(train_labels.shape))
train_labels = np.array(train_labels).astype(np.float)
# train_labels = train_labels.reshape((train_labels.shape[0],1))

test_labels = np.load('/home/sxg8458/RSNA_Bone_Age/test_labels.npy')
print('The shape of the test labels is %s '%str(test_labels.shape))
test_labels = np.array(test_labels).astype(np.float)
# test_labels = test_labels.reshape((test_labels.shape[0],1))




alphas = [0.001,0.005,0.01,0.05,0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,20.0,50.0]

with open('/home/sxg8458/RSNA_Bone_Age/output_vgg_4k.txt', 'w') as f:
    
    
    train_mae_ridge = []
    test_mae_ridge = []
    for a in alphas:
        ridge = Ridge(alpha = a)
        ridge_model = ridge.fit(train_features,train_labels)
        
        ridge_predictions_train = ridge_model.predict(train_features)
        ridge_predictions_test = ridge_model.predict(test_features)
        
        maetrain = mean_absolute_error(train_labels, ridge_predictions_train)
        maetest = mean_absolute_error(test_labels, ridge_predictions_test)
        
        train_mae_ridge.append(maetrain)
        test_mae_ridge.append(maetest)
    # print('The train mae values for ridge regression are for different lambdas are:')
    # print(train_mae_ridge)

    f.write('\n\nThe train mae for ridge regression is: \n')
    for item in train_mae_ridge:    
        f.write("%s ," % round(item,3))
    f.write('\n\nThe test mae for ridge regression is: \n')
    for item in test_mae_ridge:    
        f.write("%s ," %  round(item,3))
    
    print('#############################################ridge done#############################################')
    train_mae_lasso = []
    test_mae_lasso = []
    for a in alphas:
        lasso = Lasso(alpha = a)
        lasso_model = lasso.fit(train_features,train_labels)
        
        lasso_predictions_train = lasso_model.predict(train_features)
        lasso_predictions_test = lasso_model.predict(test_features)
        
        maetrain = mean_absolute_error(train_labels, lasso_predictions_train)
        maetest = mean_absolute_error(test_labels, lasso_predictions_test)
        
        train_mae_lasso.append(maetrain)
        test_mae_lasso.append(maetest)
    
    f.write('\n\nThe train mae for lasso regression is: \n')
    for item in train_mae_lasso:    
        f.write("%s ," %  round(item,3))
    
    f.write('\n\nThe test mae for lasso regression is: \n')
    for item in test_mae_lasso:    
        f.write("%s ," %  round(item,3))

    print('#############################################lasso done#############################################')






#################MLP########################
learning_rates = [0.001,0.005,0.01,0.05]
for lr in learning_rates:
    mlp_regressor = MLPRegressor(hidden_layer_sizes = (512,), solver = 'adam', learning_rate_init = lr, beta_1 = 0.95, activation = 'relu',max_iter = 750)
    mlp_model = mlp_regressor.fit(train_features,train_labels)
    mlp_predictions = mlp_model.predict(test_features)
    mae = mean_absolute_error(test_labels, mlp_predictions)
    print('The MAE for lr of %f for the MLP algorithm is %f ' %(lr,mae))



