#!/bin/bash
# Offers passing the owner's rules, cheapest first:  offers.sh [RTX_5090|RTX_4090]
# Rules: datacenter, verified, reliability >= 0.995, 16+ cores, inet up/down >= 300 Mbps, CUDA >= 12.8.
G=${1:-RTX_5090}
~/.local/venvs/vast/bin/vastai search offers "gpu_name=$G num_gpus=1 datacenter=True verified=True reliability>=0.995 cpu_cores_effective>=16 inet_down>=300 inet_up>=300 cuda_vers>=12.8 disk_space>=30 rentable=True" -o 'dph+' --raw \
 | python3 -c "
import json,sys
print('offer_id  \$/h  where  down up  cores  reliability')
for o in json.load(sys.stdin)[:8]: print(o['id'], round(o['dph_total'],3), o['geolocation'], round(o['inet_down']), round(o['inet_up']), round(o.get('cpu_cores_effective',0)), round(o.get('reliability2', o.get('reliability',0)),4))
"
