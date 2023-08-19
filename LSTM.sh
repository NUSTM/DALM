
#!/bin/bash
domains=('service' 'rest' 'laptop' 'device')
export CUDA_VISIBLE_DEVICES=0



for src_domain in ${domains[@]};
do
    for tar_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi

            ############# LSTM ##############
            #################################

            python ./LSTM_based/cross_domain_LM/process_data.py \
                --source_domain ${src_domain} \
                --target_domain ${tar_domain} \
                --in_path ./pseudo_outputs/ \
                --out_path ./LSTM_based/process_data/

            python ./LSTM_based/cross_domain_LM/train.py \
                --input_file ./LSTM_based/process_data/${src_domain}-${tar_domain}/final_train.txt \
                --output_dir ./LSTM_based/models/ \
                --domain_pair ${src_domain}-${tar_domain} 

            python ./LSTM_based/cross_domain_LM/generate.py \
                --target ${tar_domain} \
                --source ${src_domain} \
                --generate_number 10000

            python absa/filter.py --task absa \
                --domain_pair ${src_domain}-${tar_domain} \
                --model_name_or_path /root/data2/bert-cross/ \
                --do_filter \
                --output_dir ./LSTM_based/generated_data/
                
            python absa/main.py --task absa \
                --domain_pair ${src_domain}-${tar_domain} \
                --model_name_or_path /root/data2/bert-cross/ \
                --data_path ./LSTM_based/generated_data/${src_domain}-${tar_domain}/filter.txt \
                --output_dir ./LSTM_based/main_outputs/ \
                --n_gpu 0 \
                --do_train \
                --do_eval \
                --train_batch_size 16 \
                --eval_batch_size 16 \
                --learning_rate 3e-5 \
                --num_train_epochs 5 \
                --seed 62 


            
        fi
    done
done