from env import *
def get_spark():
    conf=SparkConf()
    conf.setExecutorEnv("processTreeMetrics","false")
    return SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option","some-value").getOrCreate()
def CFA(train_data,test_data):
    #spark=get_spark()
    def get_len(train_data,col):
        pass
    def get_mean(vals):
        df=spark.createDataFrame(pd.DataFrame({"values": vals}))
        df.createOrReplaceTempView("feature")
        res=spark.sql("SELECT AVG(values) AS mean FROM feature WHERE values IS NOT Null")
        print(res)
        print(res.toJSON())
        return json.loads(res.toJSON().first())["mean"]
    # Train
    # Student CFA
    stu_CFA={}
    for x,gr in train_data.groupby(['Anon Student Id']):
        stu_CFA[x]=len(gr[gr['Correct First Attempt']==1])
    # mean_stu_CFA=get_mean(list(stu_CFA.values()))
    # print(mean_stu_CFA)
    mean_stu_CFA=np.mean(list(stu_CFA.values()))
    # print(mean_stu_CFA)
    train_data['Student CFA']=train_data['Anon Student Id'].apply(lambda x: stu_CFA[x])

    # Problem CFA
    prob_CFA={}
    for x,gr in train_data.groupby(['Problem Name']):
        prob_CFA[x]=len(gr[gr['Correct First Attempt']==1])
    mean_prob_CFA=np.mean(list(prob_CFA.values())) # TODO: reasonable?
    train_data['Problem CFA']=train_data['Problem Name'].apply(lambda x: prob_CFA[x])

    # Step CFA
    step_CFA={}
    for x,gr in train_data.groupby(['Step Name']):
        step_CFA[x]=len(gr[gr['Correct First Attempt']==1])
    mean_step_CFA=np.mean(list(step_CFA.values())) # TODO: reasonable?
    train_data['Step CFA']=train_data['Step Name'].apply(lambda x: step_CFA[x])

    # KC CFA
    kc_CFA={}
    for x,gr in train_data.groupby(['KC(Default)']):
        kc_CFA[x]=len(gr[gr['Correct First Attempt']==1])
    mean_kc_CFA=np.mean(list(kc_CFA.values())) # TODO: reasonable?
    train_data['KC CFA']=train_data['KC(Default)'].apply(lambda x: kc_CFA[x] if not pd.isnull(x) else mean_kc_CFA)

    # Test
    # Student CFA
    test_data['Student CFA']=test_data['Anon Student Id'].apply(lambda x: stu_CFA[x] if x in stu_CFA.keys() else mean_stu_CFA)

    # Problem CFA
    test_data['Problem CFA']=test_data['Problem Name'].apply(lambda x: prob_CFA[x] if x in prob_CFA.keys() else mean_prob_CFA)

    # Step CFA
    test_data['Step CFA']=test_data['Step Name'].apply(lambda x: step_CFA[x] if x in step_CFA.keys() else mean_step_CFA)

    # KC CFA
    test_data['KC CFA']=test_data['KC(Default)'].apply(lambda x: kc_CFA[x] if x in kc_CFA.keys() else mean_kc_CFA)

    print('CFA finished.')

    return train_data,test_data

def other_features(train_data,test_data):
    # The number of KC
    train_data['KC Count']=train_data['KC(Default)'].astype("str").apply(lambda x: x.count('~~')+1 if x != 'nan' else 0)

    test_data['KC Count']=test_data['KC(Default)'].astype("str").apply(lambda x: x.count('~~')+1 if x != 'nan' else 0)

    print('Featuring finished.')

    return train_data,test_data

def encoding(train_data,test_data):
    # Train
    # Student Id encoding
    lis=list(set(train_data['Anon Student Id']).union(set(test_data['Anon Student Id'])))
    stu_enc={}
    for index,x in enumerate(lis):
        stu_enc[x]=index
    train_data['Anon Student Id']=train_data['Anon Student Id'].apply(lambda x: stu_enc[x])

    # Problem Hierarchy encoding
    lis=list(set(train_data['Problem Hierarchy']).union(set(test_data['Problem Hierarchy'])))
    hier_enc={}
    for index,x in enumerate(lis):
        hier_enc[x]=index
    train_data['Problem Hierarchy']=train_data['Problem Hierarchy'].apply(lambda x: hier_enc[x])

    # Problem Name encoding
    lis=list(set(train_data['Problem Name']).union(set(test_data['Problem Name'])))
    prob_enc={}
    for index,x in enumerate(lis):
        prob_enc[x]=index
    train_data['Problem Name']=train_data['Problem Name'].apply(lambda x: prob_enc[x])


    # Step Name encoding
    lis=list(set(train_data['Step Name']).union(set(test_data['Step Name'])))
    step_enc={}
    for index,x in enumerate(lis):
        step_enc[x]=index
    train_data['Step Name']=train_data['Step Name'].apply(lambda x: step_enc[x])

    # Test
    # Student Id encoding
    test_data['Anon Student Id']=test_data['Anon Student Id'].apply(lambda x: stu_enc[x])
    test_data['Problem Hierarchy']=test_data['Problem Hierarchy'].apply(lambda x: hier_enc[x])
    test_data['Problem Name']=test_data['Problem Name'].apply(lambda x: prob_enc[x])
    test_data['Step Name']=test_data['Step Name'].apply(lambda x: step_enc[x])

    print('Encoding finished.')

    return train_data,test_data

def output(train_data,test_data):
    features=['Anon Student Id',
            'Problem Hierarchy', 'Step Name', 'Problem Name', 'Problem View', 'Correct First Attempt', 'Student CFA', 'Problem CFA', 'Step CFA', 'KC CFA']
    train_data=train_data[features]
    test_data=test_data[features]

    train_data.to_csv('data/train_cleaned.csv',sep=',',index=False)
    test_data.to_csv('data/test_cleaned.csv',sep=',',index=False)

    print('Output finished.')

train_path='data/train.csv'
test_path='data/test.csv'
train_data=pd.read_table(train_path)
test_data=pd.read_table(test_path)
train_data,test_data=CFA(train_data,test_data)
train_data,test_data=other_features(train_data,test_data)
train_data,test_data=encoding(train_data,test_data)
output(train_data,test_data)

