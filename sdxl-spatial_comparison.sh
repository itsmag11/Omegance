prompt="A topiary plant decorated by flowers."

python sdxl_inference-controlnet_depth.py \
    -p "$prompt" \
    -o 0.0 \
    -n "o0.0"

python sdxl_inference-controlnet_depth.py \
    -p "$prompt" \
    -om \
    -oib 10.0 \
    -oif -10.0 \
    -n "mask_10.0_-10.0"

python sdxl_inference-controlnet_depth.py \
    -p "$prompt" \
    -om \
    -oib -10.0 \
    -oif 10.0 \
    -n "mask_-10.0_10.0"
