BERT_BASE_DIR='cased_L-12_H-768_A-12'

INPUT_FILE="/data2/the_beamers/data/test/test_output_daphne.json"

python extract_features.py \
  --input_file="$INPUT_FILE" \
  --output_file=/tmp/output.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layer_index=-1 \
  --max_seq_length=128 \
  --batch_size=8
