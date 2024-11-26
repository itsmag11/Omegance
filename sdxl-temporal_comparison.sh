osts=(
    'exp1'
    'exp2'
    'cos1'
    'cos2'
    )
prompt="A massive orbital station suspended against the backdrop of a planet, with intricate metallic structures, glowing docking bays, and ships entering and leaving through shimmering force fields. Inside, curved hallways are lined with sleek, glowing panels and transparent windows offering breathtaking views of the stars."

python sdxl_inference-temporal_schedules.py \
        -p "$prompt" \
        -o 1.0 \
        -n "original-o1.0"

# Loop over each prompt and pass it as an argument to the Python script
for ost in "${osts[@]}"; do

    python sdxl_inference-temporal_schedules.py \
        -p "$prompt" \
        -ost "$ost" \
        -n "$ost"

done
