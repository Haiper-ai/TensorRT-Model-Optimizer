python quantize.py \
  --model {flux-dev|sdxl-1.0|sdxl-turbo|sd2.1|sd2.1-base|sd3-medium} \
  --format int8 --batch-size 2 \
  --calib-size 32 --collect-method min-mean \
  --percentile 1.0 --alpha 0.8 \
  --quant-level 3.0 --n-steps 20 \
  --quantized-torch-ckpt-save-path ./{MODEL}_int8.pt --onnx-dir {ONNX_DIR}