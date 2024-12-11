from training_data import TrainingData
from test_data import TestData

def get_train_validation_data(): 
    print("Getting train data")
    train_data = TrainingData(512)
    #train_data.generate_new()
    #validation_data = train_data.generate_new_validation()
    train_data.generate_new_transformed(fda=False, water=False)

def get_test_data(): 
    print("Getting test data")
    test_data = TestData(512)
    test_data.generate_new()

def main(): 
    get_train_validation_data()
    get_test_data()

if __name__ == "__main__": 
    main()