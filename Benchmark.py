from config import instance
from adjointGrad_FNO import train_FNO
from adjointGrad_UNet import train_UNet
from scipy import io

trial_no = 0
reps = 1
instance.update_n_train(2500)
#instance.update_n_test(200)
instance.update_epochs(100)
for trial_no in range (reps):    
    n_train = instance.n_train
    epochs = instance.epochs
    filename_FNO = "FNO_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+ ".mat"
    filename_UNet = "UNet_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+".mat"
    total_time, test_RMSE = train_FNO()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_FNO, mdict = results_dict, format = '5')

    total_time, test_RMSE = train_UNet()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_UNet, mdict = results_dict, format = '5')


trial_no = 0
instance.update_n_train(5000)
for trial_no in range (reps):    
    n_train = instance.n_train
    epochs = instance.epochs
    print("ntrain",n_train)
    filename_FNO = "FNO_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+ ".mat"
    filename_UNet = "UNet_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+ ".mat"
    total_time, test_RMSE = train_FNO()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_FNO, mdict = results_dict, format = '5')

    total_time, test_RMSE = train_UNet()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_UNet, mdict = results_dict, format = '5')

trial_no = 0
instance.update_n_train(10000)
for trial_no in range (reps):    
    n_train = instance.n_train
    epochs = instance.epochs
    filename_FNO = "FNO_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+ ".mat"
    filename_UNet = "UNet_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+".mat"
    total_time, test_RMSE = train_FNO()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_FNO, mdict = results_dict, format = '5')

    total_time, test_RMSE = train_UNet()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_UNet, mdict = results_dict, format = '5')

    
trial_no = 0
instance.update_n_train(20000)
for trial_no in range (reps):    
    n_train = instance.n_train
    epochs = instance.epochs
    filename_FNO = "FNO_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+".mat"
    filename_UNet = "UNet_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no)+".mat"
    total_time, test_RMSE = train_FNO()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_FNO, mdict = results_dict, format = '5')

    total_time, test_RMSE = train_UNet()
    results_dict = {"Training_Time": total_time, "Test_Results": test_RMSE}
    io.savemat(file_name=filename_UNet, mdict = results_dict, format = '5')