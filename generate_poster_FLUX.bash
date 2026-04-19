# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_poster_FLUX.py \
#   --model_id /root/TOS/ZhongzhengWang/model/FLUX.1-dev \
#   --train_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_train_subset_with_farthest.csv \
#   --val_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_val_subset_with_farthest.csv \
#   --test_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_test_subset_with_farthest.csv \
#   --description_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/poster_description_subset.csv \
#   --poster_dir /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters \
#   --output_dir save/PosterGenerator \
#   --test_save_dir TestPosters \
#   --epochs 10 \
#   --train_batch_size 10 \
#   --val_batch_size 10 \
#   --test_batch_size 10 \
#   --max_description_words 100

nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_poster_FLUX.py \
  --model_id /root/TOS/ZhongzhengWang/model/FLUX.1-dev \
  --train_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_train_subset_with_farthest.csv \
  --val_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_val_subset_with_farthest.csv \
  --test_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/UserEmbeddings/user_embeddings_test_subset_with_farthest.csv \
  --description_csv /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/poster_description_subset.csv \
  --poster_dir /root/TOS/ZhongzhengWang/dataset/MovieLensLatest/MoviePosters \
  --output_dir save/PosterGenerator \
  --test_save_dir TestPosters \
  --epochs 10 \
  --train_batch_size 10 \
  --val_batch_size 10 \
  --test_batch_size 10 \
  --max_description_words 100
' > log/train.log 2>&1 &