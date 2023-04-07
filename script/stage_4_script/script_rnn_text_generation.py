from code.stage_4_code.Text_Generation_Dataset_Loader import Text_Generation_Dataset_Loader
from code.stage_4_code.Method_RNN_Text_Generation import Method_RNN_Text_Generation
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_4_code.Evaluate_Text_Generation import Evaluate_Text_Generation
import numpy as np

#---- RNN text generation script ----
if 1:
    #---- parameter section -------------------------------
    rnn_model = 'GRU' # RNN LSTM GRU
    epochs = 400
    batch_size = 128
    np.random.seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Text_Generation_Dataset_Loader('text_generation', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'

    method_obj = Method_RNN_Text_Generation('%s text generation' % rnn_model, '')
    method_obj.rnn_model = rnn_model
    method_obj.epochs = epochs
    method_obj.batch_size = batch_size

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/%s_Text_Generation_' % rnn_model
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Text_Generation('Text Generation', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    (original_texts, generated_texts), _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(rnn_model + ' text generation comparison: \n')
    for i, text in enumerate(original_texts):
        print('[%s] Original: %s' % (i, text))
        print('[%s] Generated: %s' % (i, generated_texts[i]))
    print('************ Finish ************')
    # ------------------------------------------------------
