from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader_New
from code.stage_5_code.Method_MLP import Method_MLP
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Setting import Setting_Train_Test_Data
import numpy as np
import torch
import gensim

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(10)
    torch.manual_seed(10)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    # data_obj = Dataset_Loader('stage_5', '')
    # data_obj.dataset_source_folder_path = '../../data/stage_5_data/cora/'
    # data_obj.dataset_source_link_file_name = 'link'
    # data_obj.dataset_source_node_file_name = 'node'
    data_obj = Dataset_Loader_New('Pubmed', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/pubmed/'
    data_obj.dataset_name = 'pubmed'


    method_obj = Method_MLP('MLP', '', 500, 3)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/MLP_Pubmed_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Data("Train", "Training Set")
    # in_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    