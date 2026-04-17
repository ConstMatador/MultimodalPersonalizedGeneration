CUDA_VISIBLE_DEVICES=3 nohup python -u generate_poster_description.py \
  --model_path /root/TOS/ZhongzhengWang/model/Qwen3_VL_32B \
  --input_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/info_subset.csv \
  --poster_dir /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters \
  --output_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/poster_description_subset.csv \
  --batch_size 128 \
  --dtype bfloat16 \
  --num_image_workers 16 \
  > log/generate_poster_description.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python -u generate_poster_description.py \
#   --model_path /root/TOS/ZhongzhengWang/model/Qwen3_VL_32B \
#   --input_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/info_subset.csv \
#   --poster_dir /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters \
#   --output_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/poster_description_subset.csv \
#   --batch_size 128 \
#   --dtype bfloat16 \
#   --num_image_workers 16