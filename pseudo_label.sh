
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

            python absa/pseudo_labeling.py --task absa \
                --domain_pair ${src_domain}-${tar_domain} \
                --model_name_or_path bert-cross \
                --output_dir ./pseudo_outputs/ \
                --do_train \
                --do_pseudo_labeling 

            
        fi
    done
done