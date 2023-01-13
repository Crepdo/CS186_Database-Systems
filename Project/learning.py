from env import *
def load_data(train_path,test_path):
    train_data=pd.read_table(train_path,sep=',')
    test_data=pd.read_table(test_path,sep=',')

    # Train
    X_train=train_data.dropna()
    y_train=np.array(X_train['Correct First Attempt']).astype(float).ravel()
    del X_train['Correct First Attempt']

    # Test
    X_test=test_data.dropna()
    y_test=np.array(X_test['Correct First Attempt']).astype(float).ravel()
    del X_test['Correct First Attempt']

    print('Data has been loaded.')

    return X_train,y_train,X_test,y_test

def train(X_train,y_train,X_test,y_test):
    ans=2e9
    ans_model=None
    # Decision Tree
    model=tree.DecisionTreeClassifier().fit(X_train,y_train)
    y_pred=model.predict(X_test)
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'Decision Tree: {RMSE}')

    '''
    # Random Forest
    model=RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=None,min_samples_split=0.01,
                                 n_jobs=-1,random_state=None,verbose=0).fit(X_train,y_train)
    # y_pred=model.predict_proba(X_test)[:,1]
    y_pred=model.predict(X_test)
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'Random Forest: {RMSE}')

    # AdaBoost
    model=AdaBoostRegressor(base_estimator=None,n_estimators=200,learning_rate=0.1,
                            loss='exponential',random_state=None).fit(X_train,y_train)
    y_pred=model.predict(X_test)
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'AdaBoost: {RMSE}')

    '''
    # Logistic Regression
    model=LogisticRegression(penalty='l2',max_iter=200).fit(X_train,y_train)
    # y_pred=model.predict_proba(X_test)[:,1]
    y_pred=model.predict(X_test)
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'Logistic Regression: {RMSE}')
    
    '''
    # GBDT
    model=GradientBoostingClassifier(n_estimators=200).fit(X_train,y_train)
    # y_pred=model.predict_proba(X_test)[:,1]
    y_pred=model.predict(X_test)
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'GBDT: {RMSE}')'''

    # LightGBM
    model=lightgbm.LGBMClassifier(boosting_type='gbdt',objective='binary',max_depth=5,num_leaves=20,learning_rate=0.1,n_estimators=1000,n_jobs=-1,
                               min_child_samples=20,subsample=0.85,subsample_freq=1,boost_from_average=False,reg_lambda=0.1,verbose=-1).fit(X_train,y_train)
    # y_pred=model.predict_proba(X_test)[:,1]
    y_pred=model.predict(X_test)
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'LightGBM: {RMSE}')

    # LSTM
    model=genLSTM(len(X_train.columns))
    history=model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train','validation'],loc='upper right')
    plt.show()
    y_pred=model.predict(X_test).round()
    RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
    if RMSE<ans:
        ans=RMSE
        ans_model=model
    print(f'LSTM: {RMSE}')


    print(f'Best Result: {ans}')
    return ans_model

def predict(predict_ori_path,test_path,model):
    # No dropna to predict
    test_data=pd.read_table(test_path,sep=',')
    X_test=test_data
    y_test=np.array(X_test['Correct First Attempt']).astype(float).ravel()
    del X_test['Correct First Attempt']

    #y_pred=model.predict_proba(X_test)[:,1]
    y_pred=model.predict(X_test).round()
    for i,x in enumerate(y_test):
        if np.isnan(x):
            y_test[i]=y_pred[i]
    res=pd.read_table(predict_ori_path,sep='\t')
    res['Correct First Attempt']=y_test
    res.to_csv('result/test.csv',sep='\t',index=False)
    print('Result has been outputted.')

train_path='data/train_cleaned.csv'
test_path='data/test_cleaned.csv'
predict_ori_path='data/test.csv'
X_train,y_train,X_test,y_test=load_data(train_path,test_path)
ans_model=train(X_train,y_train,X_test,y_test)
predict(predict_ori_path,test_path,ans_model)
