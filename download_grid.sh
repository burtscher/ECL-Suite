#!/bin/bash
# This script is used to download the input files for APSP to the inputs-grid directory

mkdir -p inputs-grid

wget -nc -P inputs-grid --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/GD99_b.egr
wget -nc -P inputs-grid --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/celegansneural.egr
wget -nc -P inputs-grid --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/jpwh_991.egr
wget -nc -P inputs-grid --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/CollegeMsg.egr
wget -nc -P inputs-grid --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/soc-sign-bitcoin-alpha.egr

