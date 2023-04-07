from code.stage_4_code.Text_Classification_Dataset_Loader import Text_Classification_Dataset_Loader
from code.stage_4_code.Method_RNN_Text_Classification import Method_RNN_Text_Classification
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test_Already_Split import Setting_Train_Test_Already_Split
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np

#---- RNN text classification script ----
if 1:
    #---- parameter section -------------------------------
    rnn_model = 'LSTM' # RNN LSTM GRU
    epochs = 10
    batch_size = 128
    np.random.seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Text_Classification_Dataset_Loader('text_classification', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'

    method_obj = Method_RNN_Text_Classification('%s text classification' % rnn_model, '')
    method_obj.rnn_model = rnn_model
    method_obj.epochs = epochs
    method_obj.batch_size = batch_size

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/%s_Text_Classification_' % rnn_model
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Already_Split('train test already split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(rnn_model + ' text classification Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
