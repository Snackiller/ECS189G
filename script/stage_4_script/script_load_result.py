from code.stage_4_code.Result_Loader import Result_Loader

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/GRU_Text_Generation_'
    result_obj.result_destination_file_name = 'prediction_result'

    for fold_count in [None]:
        result_obj.fold_count = fold_count
        result_obj.load()
        print('Fold:', fold_count, ', Result:', result_obj.data)