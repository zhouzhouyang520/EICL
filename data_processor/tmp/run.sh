#!/bin/sh

out_file=out.log

#nohup python emotion_probability_generator.py --batch_size 1000 > $out_file 2>&1 &

#nohup python build_data.py --cuda --device_id 4 --emb_model emotion --batch_size 512 > $out_file 2>&1 &

nohup python chatgpt.py > $out_file 2>&1 &

tail -f $out_file
