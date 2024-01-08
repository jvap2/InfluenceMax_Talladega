#!/bin/sh
# export vers=3
# export seed_size=100
# export ep=.5

sh nd_run_IMM.sh
sh amazon_run_IMM.sh
sh epinions_run.sh
sh run_IMM.sh
sh arvix_run_IMM.sh
sh wikivote_run_IMM.sh
sh hepth_run_IMM.sh