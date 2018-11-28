### VGG Features
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

#################Load the train data########################

train_features = np.load('/home/sxg8458/RSNA_Bone_Age/train_images_vgg_4k.npy')
print('The shape of the train features is %s ' %str(train_features.shape))


test_features = np.load('/home/sxg8458/RSNA_Bone_Age/test_images_vgg_4k.npy')
print('The shape of the test features is %s ' %str(test_features.shape))


train_labels = np.load('/home/sxg8458/RSNA_Bone_Age/train_labels.npy')
print('The shape of the train labels is %s '%str(train_labels.shape))
train_labels = np.array(train_labels).astype(np.float)
# train_labels = train_labels.reshape((train_labels.shape[0],1))

test_labels = np.load('/home/sxg8458/RSNA_Bone_Age/test_labels.npy')
print('The shape of the test labels is %s '%str(test_labels.shape))
test_labels = np.array(test_labels).astype(np.float)
# test_labels = test_labels.reshape((test_labels.shape[0],1))
'''

#################Linear Regression########################
lm = LinearRegression()
lm_model = lm.fit(train_features,train_labels)
lm_predictions = lm_model.predict(test_features)


mae = mean_absolute_error(test_labels, lm_predictions)
print('The MAE for the linear regression is %f ' %mae)




#################Ridge Regression########################
ridge = Ridge(alpha = 5.0)
ridge_model = ridge.fit(train_features,train_labels)
ridge_predictions = ridge_model.predict(test_features)

mae = mean_absolute_error(test_labels, ridge_predictions)
print('The MAE for the ridge regression is %f ' %mae)
# print(np.hstack((test_labels,ridge_predictions))[0:20])





#################Lasso Regression########################
lasso = Lasso(alpha = 0.01)
lasso_model = lasso.fit(train_features,train_labels)
lasso_predictions = lasso_model.predict(test_features)

mae = mean_absolute_error(test_labels, lasso_predictions)
print('The MAE for the lasso regression is %f ' %mae)
print(lasso_predictions.shape)
# print(np.hstack((test_labels,lasso_predictions.reshape((lasso_predictions.shape[0],1))))[0:20])


#################Elastic Net Regression########################
elnet = ElasticNet(alpha=1.0, l1_ratio=0.5)
elnet_model = elnet.fit(train_features,train_labels)
elnet_predictions = elnet_model.predict(test_features)

mae = mean_absolute_error(test_labels, elnet_predictions)
print('The MAE for the elnet regression is %f ' %mae)
# print(np.hstack((test_labels,elnet_predictions))[0:20])



#################Support Vector Regression########################
svm = SVR(kernel = 'rbf',C = 0.5)
svm_model = svm.fit(train_features,train_labels)
svm_predictions = svm_model.predict(test_features)
mae = mean_absolute_error(test_labels, svm_predictions)
print('The MAE for SVM Regression is : %f' %mae)


#################MLP########################
mlp_regressor = MLPRegressor(hidden_layer_sizes = (512,), solver = 'adam', learning_rate_init = 0.01, beta_1 = 0.95, activation = 'relu')
mlp_model = mlp_regressor.fit(train_features,train_labels)
mlp_predictions = mlp_model.predict(test_features)
mae = mean_absolute_error(test_labels, mlp_predictions)
print('The MAE for MLP is %f ' %mae)



'''

#################GBM ########################
gbm = GradientBoostingRegressor(loss = 'ls',learning_rate=0.01, n_estimators=500,min_samples_split=4, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5)
gbm_model = gbm.fit(train_features,train_labels)
gbm_predictions = gbm_model.predict(test_features)
mae = mean_absolute_error(test_labels, gbm_predictions)
print('The MAE for GBM Regression is : %f' %mae)



#################XgBoost ########################
xgboost = xgb.XGBRegressor()
xgboost_model = xgboost.fit(train_features,train_labels)
xgboost_predictions = xgboost_model.predict(test_features)
mae = mean_absolute_error(test_labels, xgboost_predictions)
print('The MAE for xgboost Regression is : %f' %mae)