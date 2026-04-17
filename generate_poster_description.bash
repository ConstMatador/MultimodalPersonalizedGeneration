# CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_poster_description.py \
#   --model_path /root/TOS/ZhongzhengWang/model/Qwen3_VL_32B \
#   --input_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/info_subset.csv \
#   --poster_dir /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters \
#   --output_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/poster_description_subset.csv \
#   --batch_size 1024 \
#   --dtype bfloat16

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u generate_poster_description.py \
  --model_path /root/TOS/ZhongzhengWang/model/Qwen3_VL_32B \
  --input_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/info_subset.csv \
  --poster_dir /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters \
  --output_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/poster_description_subset.csv \
  --batch_size 8 \
  --dtype bfloat16 \
  > log/generate_poster_description.log 2>&1 &