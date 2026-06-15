declare -A forecasting_length
declare -A dataset_text

keys=("COVID_Deaths" "NN5_Daily" "Car_Parts" "Australian_Electricity" "CIF_2016" "Dominick" "Hospital" "Tourism_Monthly")

forecasting_length=(
    [COVID_Deaths]=30
    [NN5_Daily]=56
    [Car_Parts]=12
    [Australian_Electricity]=48
    [CIF_2016]=12
    [Dominick]=8
    [Hospital]=12
    [Tourism_Monthly]=24
)

dataset_text=(
    [COVID_Deaths]="This dataset contains daily time series that represent the COVID-19 deaths in a set of countries and states"
    [NN5_Daily]="This dataset was used in the NN5 forecasting competition and contains the daily cash withdrawals from ATMs in UK"
    [Car_Parts]="This dataset contains monthly sales data for various car parts, measured between January 1998 and March 2002"
    [Australian_Electricity]="This dataset contains the half hourly electricity demand of 5 states in Australia"
    [CIF_2016]="This dataset contains monthly banking data that was used in the CIF 2016 forecasting competition"
    [Dominick]="This dataset contains weekly time series representing the profit of individual stock keeping units from a retailer"
    [Hospital]="This dataset contains monthly time series that represent the patient counts related to medical products from January 2000 to December 2006"
    [Tourism_Monthly]="Tourism dataset contains monthly time series used in the Kaggle Tourism forecasting competition"
)

vision_model=""
text_model=""
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model="CLIP"
text_model=""
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model="BLIP"
text_model=""
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model=""
text_model="Qwen"
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    dt="${dataset_text[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --dataset_text "$dt"\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model=""
text_model="LLaMA"
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    dt="${dataset_text[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --dataset_text "$dt"\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model="CLIP"
text_model="Qwen"
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    dt="${dataset_text[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --dataset_text "$dt"\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model="CLIP"
text_model="LLaMA"
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    dt="${dataset_text[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --dataset_text "$dt"\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model="BLIP"
text_model="Qwen"
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    dt="${dataset_text[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --dataset_text "$dt"\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done

vision_model="BLIP"
text_model="LLaMA"
echo -e "\n\n"$vision_model""$text_model"Chronos"
for key in "${keys[@]}"; do
    fl="${forecasting_length[$key]}"
    dt="${dataset_text[$key]}"
    echo -e "\n$key"
    python test_multi_modal_chronos.py \
        --forecasting_length $fl \
        --test_dataset_path /home/ssh_adnlp/TSF/Vision_TSFM/dataset/$key/test\
        --dataset_text "$dt"\
        --checkpoint_path /home/ssh_adnlp/TSF/Vision_TSFM/ckpt/"$vision_model""$text_model"Chronos/$key
done