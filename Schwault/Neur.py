

def NN(test_dataset):

    #Artificial Neural Network

    # Importing the libraries
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    str1 = "./static/"
    td = str1 + test_dataset

    # Part 1 - Data Preprocessing

    # Importing the dataset
    dataset = pd.read_csv("fraudTrain_final.csv")
    dataset_test = pd.read_csv(td)
    dataset = dataset.drop(columns=['merchant','street','job','dob','trans_num','Unnamed: 0','city_pop'])
    dataset_test = dataset_test.drop(columns=['merchant','street','job','dob','trans_num','Unnamed: 0','city_pop'])
    dataset. columns = dataset. columns. str. replace(' ','_')
    dataset_test. columns = dataset_test. columns. str. replace(' ','_')

    print(dataset.axes)
    print("**********")
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values

    Xt = dataset_test.iloc[:, 0:-1].values
    yt = dataset_test.iloc[:, -1].values


    print("Test Dataset")
    print("************************")
    print(Xt)
    print("***************")
    print(yt)

    # Encoding categorical data
    def l_encode(dataset, X):
        # Label Encoding the "Gender" column
        print("*******************")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 6] = le.fit_transform(X[:, 6])


        # Label Encoding the "city" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 7] = le.fit_transform(X[:, 7])



        # Label Encoding the "category" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])




        # Label Encoding the "state" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 8] = le.fit_transform(X[:, 8])




        # Label Encoding the "time and date" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 0] = le.fit_transform(X[:, 0])




        # Label Encoding the "first name" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 4] = le.fit_transform(X[:, 4])

        # Label Encoding the "last name" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 5] = le.fit_transform(X[:, 5])
        print(X)


        return dataset

    dataset = l_encode(dataset, X)
    print("\nTesting DS\n")
    dataset_test = l_encode(dataset_test, Xt)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    #For test dataset
    from sklearn.preprocessing import StandardScaler
    sc1 = StandardScaler()
    Xt = sc1.fit_transform(Xt)
    print(Xt)

    # Part 2 - Building the ANN

    # Initializing the ANN
    ann = tf.keras.models.Sequential()

    # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the third hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the fourth hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    print("Built")


    # Part 3 - Training the ANN

    # Compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training the ANN on the Training set
    ann.fit(X_train, y_train, batch_size = 60, epochs = 10)


    # Predicting the Test set results
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)


    # Predicting the Test set results
    yt_pred = ann.predict(Xt)
    yt_pred = (yt_pred > 0.5)
    #print(np.concatenate((yt_pred.reshape(len(yt_pred),1), yt.reshape(len(yt),1)),1))

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm1 = confusion_matrix(yt, yt_pred)
    accuracy_score(yt, yt_pred)

    list1 = [yt_pred]
    list1 = list1[0]

    dataset_test['PREDICTION'] = list1
    dataset_test.to_csv('./static/result2.csv')
    return dataset_test
