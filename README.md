
KNN : Accuracy = 71.0498%

RANDOM FOREST: 70.41% Accuracy

AdaBoost :  Accuracy = 56%

Best Model: KNN with 71.0498% Accuracy



RANDOM FOREST: 70.41% Accuracy forest_model.txt is trained highly and precisely for best splits on training data from value of splits ranging from 0 to 255 for each column. It took 3 hours to train this model. So, in order to let AI's check whether my code is working or not, I have uploaded code of random forest training such that it achieves accuracy ranging from 63.5% to 67.2% if you train model and than run that model.
  
    Thus: Random Forest:   for testing:    forest_model.txt ---> 70.41% Accuracy ----> 3 hours to train 
                           for training :  For ease of AI's Code Working or not checking: 63.5% to 67.2% accuracy----> will take 30 seconds hardly to run
                           
                           IF AI's want model with 70.41% Accuracy in training also(Though after knowing it will take 3-4 hours or more time ), then please comment line-----> for c in [128]:    
                           and uncomment line --------->  # for c in range(1,255):
                           
                           1st line was just splitting on 128 for each column, while second line would split on 1 to 255 values for each column.


