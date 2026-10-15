tune_model() {
  local mode="$1"
  #local icsv="aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_${m}.csv"
  #local ocsv="/tmp/a8w8_blockscale_tuned_fmoe_${m}.csv"
  local icsv="$2"
  local ocsv="$3"
  local a="-i ${icsv} -o ${ocsv} --mp 1 --warmup 10 --iters 51 --timeout 180 --all"
  local T=csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
  (
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=1 TUNE_ONLY=flydslv2,cktile,flydsl,asm 
  case "$mode" in
  tune) python3 "$T" $a ;;
  e2e) python3 "$T" $a --e2e_tune ;;
  run) python3 "$T" $a --run_config "$ocsv" ;;
  *) echo "unknown mode: $mode" >&2; return 1 ;;
  esac
  echo python3 "$T" $a
  )
}



# aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_mimo_v2_6_pro_rl.csv

echo tune_model tune 0.csv 1.csv
echo cp 1.csv 2.csv
echo tune_model e2e 0.csv 2.csv

