from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.applications import ResNet50#load resnet 50 as transfer learning
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras_applications.resnext import ResNeXt50 #load resnext50 as propose transfer learning
import keras
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

global labels
global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
global accuracy, precision, recall, fscore, values,cnn_model, text
global ensemble_model
global resnext_model
precision = []
recall = []
fscore = []
accuracy = []

main = tkinter.Tk()
main.title("Autonomous Landing Scene Recognition") #designing main screen
main.geometry("1300x1200")

path = "LandingDataset"
labels = []
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())
print("Landing Scenes 7 Dataset Class Labels : "+str(labels)) 

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, dataset, labels, X_train, Y_train, text
    text.delete('1.0', END)
    global filename
    global X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists('model/X.txt.npy'): #load dataset from processed models
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else: #dataset not processed then read and save and load
        X = []
        Y = []
        for root, dirs, directory in os.walk(path): #loop all dataset images
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])#read image
                    img = cv2.resize(img, (32,32))#resize image
                    X.append(img) #add image features to training X
                    label = getLabel(name) #get image label
                    Y.append(label) #add label to Y array
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save('model/X.txt',X)#save processed images
    np.save('model/Y.txt',Y)                    
    unique, count = np.unique(Y, return_counts=True)   
    text.insert(END,"Dataset loading task completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()
    
def processDataset():
    global dataset,X,Y,values
    global X_train, X_test, y_train, y_test, pca, scaler
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255 #normalizing image training features
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Processing Completed\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    print()
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.xticks(rotation=90)
    plt.show()

def Resnext():
    global resnext_model
    text.delete('1.0', END)
    resnext = ResNeXt50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), backend = keras.backend,
                layers = keras.layers, models = keras.models, utils = keras.utils)
    for layer in resnext.layers:
        layer.trainable = False
    resnext_model = Sequential()
    resnext_model.add(resnext)
    resnext_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnext_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnext_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    resnext_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnext_model.add(Flatten())
    resnext_model.add(Dense(units = 256, activation = 'relu'))
    resnext_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    resnext_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnext_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnext_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnext_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnext_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        resnext_model = load_model("model/resnext_weights.hdf5")
        #perform prediction on test data   
        predict = resnext_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        test = np.argmax(y_test, axis=1)
        calculateMetrics("Propose ResNext50 with ADAM", predict, test)

def Resnet():
    text.delete('1.0', END)
    #now train ResNet50 as existing algorithm
    #define resnet50 object
    resnet = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    resnet_model = Sequential()
    #add resnet50 object to CNN as tranfer learning
    resnet_model.add(resnet)
    #define parameters for transfer learning
    resnet_model.add(GlobalAveragePooling2D())
    resnet_model.add(Dense(2, activation='relu'))
    resnet_model.add(BatchNormalization())
    resnet_model.add(Dropout(0.2))
    #define prediction output layer
    resnet_model.add(Dense(y_train.shape[1], activation='sigmoid'))
    #compile and train the model
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        resnet_model = load_model("model/resnet_weights.hdf5")
        #perform prediction on test data using resnet tranfer learning model   
        predict = resnet_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        test = np.argmax(y_test, axis=1)
        calculateMetrics("Existing ResNet50 with ADAM", predict, test)

def EnsembleRandomForest():
    global resnext_model, Y,ensemble_model,rf
    text.delete('1.0', END)
    #now train hybrid ensemble random forest algorithm as extension by extracting features from trained ResNext50 model and then
    #retrain extracted features using Random Forest to build hybrid ensemble model and then comprae its accuracy with propose model
    ensemble_model = Model(resnext_model.inputs, resnext_model.layers[-2].output)#creating hybrid model object using ResNext
    ensemble_features = ensemble_model.predict(X)  #extracting ResNext features from dataset X
    Y = np.argmax(Y, axis=1)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(ensemble_features, Y, test_size=0.2) #split dataset into train and test
    #now train random forest on hybrid features
    rf = RandomForestClassifier()
    rf.fit(ensemble_features, Y)
    #perform prediction on test data
    predict = rf.predict(X_test)
    calculateMetrics("Extension Hybrid Ensemble Random Forest", predict, y_test)

def graph():
    # Check if metrics are available
    if len(precision) < 3 or len(recall) < 3 or len(fscore) < 3 or len(accuracy) < 3:
        text.insert(END, "Error: Metrics for models are not available. Make sure you have run all models first.\n")
        return

    # Construct the DataFrame for plotting
    df = pd.DataFrame([
        ['Existing ResNet50', 'Precision', precision[1]],
        ['Existing ResNet50', 'Recall', recall[1]],
        ['Existing ResNet50', 'F1 Score', fscore[1]],
        ['Existing ResNet50', 'Accuracy', accuracy[1]],
        ['Propose ResNext50', 'Precision', precision[0]],
        ['Propose ResNext50', 'Recall', recall[0]],
        ['Propose ResNext50', 'F1 Score', fscore[0]],
        ['Propose ResNext50', 'Accuracy', accuracy[0]],
        ['Extension ResNext50 + Random Forest Hybrid Model', 'Precision', precision[2]],
        ['Extension ResNext50 + Random Forest Hybrid Model', 'Recall', recall[2]],
        ['Extension ResNext50 + Random Forest Hybrid Model', 'F1 Score', fscore[2]],
        ['Extension ResNext50 + Random Forest Hybrid Model', 'Accuracy', accuracy[2]],
    ], columns=['Parameters', 'Algorithms', 'Value'])
    
    # Pivot the data for better visualization
    pivot_df = df.pivot("Parameters", "Algorithms", "Value")
    
    # Plot the bar chart
    pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title("All Algorithms Performance Comparison")
    plt.ylabel('Scores')
    plt.xlabel('Metrics')
    plt.show()


def accuracygraph():
    f = open('model/resnext_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['val_acc']
    loss = data['val_loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss Rate')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['ResNext50 Accuracy', 'ResNext50 Loss'], loc='upper left')
    plt.title('ResNext50 Training Accuracy & Loss Graph')
    plt.show()

def predictLanding():
    global filename, dataset,ensemble_model,rf,labels
    text.delete('1.0', END)
    labels=['Building', 'Field', 'Lawn', 'Mountain', 'Road', 'Vehicles', 'WaterArea', 'Wilderness']
    filename=filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = img.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255 #normalizing test image
    predict = ensemble_model.predict(img) #using ensemble ResNext50 model we are extracting features from given image
    predict = rf.predict(predict)
    predict = predict[0]
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Landing Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    plt.figure(figsize=(8,8))
    cv2.imshow('Image Classified as : '+labels[predict], img)
    cv2.waitKey(0)

font = ('times', 16, 'bold')
title = Label(main, text='Autonomous Landing Scene Recognition Based on Transfer Learning For Drones')
title.config(bg='white', fg='purple')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=250,y=100)
processButton.config(font=font1)

resnextButton = Button(main, text="Run Proposed ResNext50", command=Resnext)
resnextButton.place(x=500,y=100)
resnextButton.config(font=font1)

resnetButton = Button(main, text="Run Existing ResNet50", command=Resnet)
resnetButton.place(x=750,y=100)
resnetButton.config(font=font1)

ensembleButton = Button(main, text="Run Extension Ensemble Random Forest", command=EnsembleRandomForest)
ensembleButton.place(x=1000,y=100)
ensembleButton.config(font=font1)

graphButton = Button(main, text="Comparision Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

agraphButton = Button(main, text="ResNext50 Accuracy Graph", command=accuracygraph)
agraphButton.place(x=250,y=150)
agraphButton.config(font=font1)

predictButton = Button(main, text="Predict from Test Image", command=predictLanding)
predictButton.place(x=500,y=150)
predictButton.config(font=font1)

main.config(bg='purple')
main.mainloop()
