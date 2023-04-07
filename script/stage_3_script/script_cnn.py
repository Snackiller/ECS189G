from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_MNIST import Setting_Train_Test_MNIST
from code.stage_3_code.Setting_Train_Test_ORL import Setting_Train_Test_ORL
from code.stage_3_code.Setting_Train_Test_CIFAR import Setting_Train_Test_CIFAR
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

name = "CIFAR"

if name == "MNIST":
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train/test', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'MNIST'

    method_obj = Method_CNN_MNIST('CNN_MNIST', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_MNIST_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_MNIST('Train and Test Setting', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------


if name == "ORL":
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train/test', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN_ORL('CNN_ORL', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_ORL('Train and Test Setting', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------

if name == "CIFAR":
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train/test', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN_CIFAR('CNN_CIFAR', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_CIFAR('Train and Test Setting', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    