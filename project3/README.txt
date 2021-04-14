###     Torch Version     ####

pytorch version: 1.4.0


###     Training args      ####

epochs: 15
learning rate: .001
batch size: 4
optimization method: Cross Entropy


###    Classification testing accuracy overall and class-wise    ####

Accuracy of the network on the 10000 test images: 64 %

Accuracy of plane : 72 %
Accuracy of   car : 70 %
Accuracy of  bird : 41 %
Accuracy of   cat : 42 %
Accuracy of  deer : 62 %
Accuracy of   dog : 55 %
Accuracy of  frog : 85 %
Accuracy of horse : 53 %
Accuracy of  ship : 79 %
Accuracy of truck : 79 %

###      AUC result      ###


For some reason OTB2013.json was bugged for me so I had to change the json path to OTB2015.json.
I don't think it should affect the result but just a forewarning for reference. 


OTB2013 Best: result\OTB2013\DCFNet_test(0.5348)