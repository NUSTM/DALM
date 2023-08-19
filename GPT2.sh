
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

            ############ GPT-2 #############
            ################################

            python ./GPT2_based/cross_domain_LM/process_data.py \
                --source_domain ${src_domain} \
                --target_domain ${tar_domain} \
                --in_path ./pseudo_outputs/ \
                --out_path ./GPT2_based/process_data/

            python ./GPT2_based/cross_domain_LM/train.py \
                --input_dir ./GPT2_based/process_data/${src_domain}-${tar_domain}/final_train.txt \
                --model_dir ./GPT2_based/models/ \
                --source_domain ${src_domain} \
                --target_domain ${tar_domain} 

            python ./GPT2_based/cross_domain_LM/generate.py \
                --target ${tar_domain} \
                --source ${src_domain} \
                --generate_number 10000

            python absa/filter.py --task absa \
                --domain_pair ${src_domain}-${tar_domain} \
                --model_name_or_path /root/data2/bert-cross/ \
                --do_filter \
                --output_dir ./GPT2_based/generated_data/
                
            python absa/main.py --task absa \
                --domain_pair ${src_domain}-${tar_domain} \
                --model_name_or_path /root/data2/bert-cross/ \
                --data_path ./GPT2_based/generated_data/${src_domain}-${tar_domain}/filter.txt \
                --output_dir ./GPT2_based/main_outputs/ \
                --do_train \
                --do_eval 

            
        fi
    done
done