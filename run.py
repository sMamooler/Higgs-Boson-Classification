import numpy as np
from proj1_helpers import *
from implementations import *
from cross_validation import *



#----------------------------------------LOADING------------------------------------------------------------------------------
print("Loading data:")
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, raw_tX, ids = load_csv_data(DATA_TRAIN_PATH)

#split the data into training and validation sets
x_train, y_train, ids_train, x_val, y_val, ids_val = split_data(raw_tX, y, ids, 0.8)
y = np.array(y)
raw_tX = np.array(raw_tX)
print("Loading finished!")

jets = [0, 1, 2, 3]
jet_indx = [None] * len(jets)
jet_y = [None] * len(jets)
jet_data = [None] * len(jets)
jet_id = [None] * len(jets)
tX = [None] * len(jets)



#---------------------------------------PRE-PROCESSING-----------------------------------------------------------------------
print("Pre-processing:")
for jet_nb in jets:
    jet_indx[jet_nb], jet_y[jet_nb], jet_data[jet_nb], jet_id[jet_nb] = partition(y_train, x_train, ids_train, jet_nb)
    tX[jet_nb], _= preprocessing(jet_data[jet_nb])
print("Pre-processing finished!")


weights =[None] * len(jets)
loss = [None] * len(jets)
degrees = [None] * len(jets)
lambdas = [None] * len(jets)



#uncomment for tuning
#-----------------------------------------TUNING------------------------------------------------------------------------------
#print("Tuning:")
#for jet_nb in jets:
    #last argument to be changed
#    lambdas[jet_nb], degrees[jet_nb] = k_fold_cross_validation(jet_y[jet_nb], tX[jet_nb], 5, orig_features[jet_nb])
#    print("optimal lambda for jet {i}: {lamb}".format(i=jet_nb, lamb=lambdas[jet_nb]))
#    print("optimal degree for jet {i}: {deg}".format(i=jet_nb, deg=degrees[jet_nb]))

#print("Tuning finished!")


degrees = [6,6,6,6]
lambdas = [4.175318936560409*(10**(-12)), 5.736152510448681*(10**(-8)), 1.7433288221999908*(10**(-8)), 1.373823795883261*(10**(-11))  ]
#------------------------------------------TRAINING---------------------------------------------------------------------------
for jet_nb in jets:
    tX[jet_nb] = build_poly(tX[jet_nb], degrees[jet_nb])
    weights[jet_nb],_ = ridge_regression(jet_y[jet_nb], tX[jet_nb], lambdas[jet_nb])
    loss[jet_nb] = compute_loss(jet_y[jet_nb], tX[jet_nb], weights[jet_nb])
    print("Best loss for jet {i}: {loss}".format(i=jet_nb, loss = loss[jet_nb]))

#-------------------------------------VALIDATING--------------------------------------------------------------------------
print("Validating:")
val_indx = [None] * len(jets)
val_y = [None] * len(jets)
val_data = [None] * len(jets)
val_id = [None] * len(jets)
val_tX = [None] * len(jets)
val_preds = [None] * len(jets)

#partition, pre.process and predict the weights
for jet_nb in jets:
    val_indx[jet_nb], val_y[jet_nb], val_data[jet_nb], val_id[jet_nb] = partition(y_val, x_val, ids_val, jet_nb)
    val_tX[jet_nb], _ = preprocessing(val_data[jet_nb])
    
    val_tX[jet_nb] = build_poly(val_tX[jet_nb], degrees[jet_nb])
    val_preds[jet_nb] = predict_labels(weights[jet_nb], val_tX[jet_nb])

#merge all aprtitions:
y_pred_all = []
ids_all = []
y_all = []
for id in val_id:
    ids_all =  np.concatenate((ids_all, id))
for p in val_preds:
    y_pred_all = np.concatenate((y_pred_all, p))
for e in val_y:
    y_all = np.concatenate((y_all, e))

#sort according to ids
sorted_indx_val = np.argsort(ids_all)
ids_all = ids_all[sorted_indx_val]
y_pred_all = y_pred_all[sorted_indx_val]
y_all = y_all[sorted_indx_val]
#compute accuracy
acc = compute_accuracy(y_pred_all, y_all)
print("Accuracy: {a}.".format(a=acc))

#---------------------------------------------------PREDICTING--------------------------------------------------------------------
print("Predicting")
DATA_TEST_PATH = 'test.csv'
y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)
test_indx = [None] * len(jets)
test_y = [None] * len(jets)
test_data = [None] * len(jets)
test_id = [None] * len(jets)
test_tX = [None] * len(jets)
preds = [None] * len(jets)


#partition, pre-process and predict the weights

for jet_nb in jets:
    test_indx[jet_nb], test_y[jet_nb], test_data[jet_nb], test_id[jet_nb] = partition(y_test, x_test, ids_test, jet_nb)
    test_tX[jet_nb], _ = preprocessing(test_data[jet_nb]) 
    
    test_tX[jet_nb] = build_poly(test_tX[jet_nb], degrees[jet_nb])
    preds[jet_nb] = predict_labels(weights[jet_nb], test_tX[jet_nb])

#merge all partitions:
y_pred = []
ids_all_pred = []
for id in test_id:
    ids_all_pred =  np.concatenate((ids_all_pred, id))
for p in preds:
    y_pred = np.concatenate((y_pred, p))

#sort according to ids
sorted_indx = np.argsort(ids_all_pred)
ids_all_pred = ids_all_pred[sorted_indx]
y_pred = y_pred[sorted_indx]

OUTPUT_PATH = 'output.csv'
create_csv_submission(ids_all_pred, y_pred, OUTPUT_PATH)
print("Done!")
