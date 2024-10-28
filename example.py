import sagemaker
from sklearn.model_selection import train_test_split
import boto3
import pandas as pd

region = 'us-east-2'

boto3_session = boto3.Session(region_name=region)
sm_boto3 = boto3_session.client("sagemaker")
sess = sagemaker.Session(boto_session=boto3_session)
region = sess.boto_session.region_name

bucket = 'mobbucketsagemakerjonah'

df = pd.read_csv("mob_price_classification_train.csv")

# print(df.head())
# print(df.shape)

df['price_range'].value_counts(normalize=True)

# print(df.columns)

features = list(df.columns)
label = features.pop(-1)

x = df[features]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=0)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

trainX = pd.DataFrame(X_train)
trainX[label] = y_train

testX = pd.DataFrame(X_test)
testX[label] = y_test

trainX.to_csv("train-V-1.csv",index = False)
testX.to_csv("test-V-1.csv", index = False)

# send data to S3. SageMaker will take training data from s3
sk_prefix = "sagemaker/mobile_price_classification/sklearncontainer"
trainpath = sess.upload_data(
    path="train-V-1.csv", bucket=bucket, key_prefix=sk_prefix
)

testpath = sess.upload_data(
    path="test-V-1.csv", bucket=bucket, key_prefix=sk_prefix
)
# print(trainpath)
# print(testpath)



from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role="arn:aws:iam::207567791989:role/service-role/AmazonSageMaker-ExecutionRole-20240911T125168",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="RF-custom-sklearn",
    hyperparameters={
        "n_estimators": 100,
        "random_state": 0,
    },
    use_spot_instances = True,
    max_wait = 7200,
    max_run = 3600
)




# # launch training job, with asynchronous call
sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=True)
# # sklearn_estimator.fit({"train": datapath}, wait=True)




sklearn_estimator.latest_training_job.wait(logs="None")
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]

# print("Model artifact persisted at " + artifact)



from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime

model_name = "1Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model = SKLearnModel(
    name =  model_name,
    model_data=artifact,
    role="arn:aws:iam::207567791989:role/service-role/AmazonSageMaker-ExecutionRole-20240911T125168",
    entry_point="script.py",
    framework_version=FRAMEWORK_VERSION,
)



##Endpoints deployment
endpoint_name = "2Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName={}".format(endpoint_name))

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
)

testX[features][0:2].values.tolist()
print(predictor.predict(testX[features][0:2].values.tolist()))
sm_boto3.delete_endpoint(EndpointName=endpoint_name)