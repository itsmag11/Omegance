floats=(-5.0 0.0 5.0)
prompt="A snowy mountain peak where a family of polar bears rests on an icy ledge, surrounded by glistening icicles and a stunning aurora borealis lighting up the night sky."

# Loop over each integer
for i in "${floats[@]}"; do

    python sdxl_inference.py \
        -p "$prompt" \
        -o "$i" \
        -n "o$i"

done