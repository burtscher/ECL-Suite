#!/bin/bash
# This script is used to download the input files to the input directories
# Warning: requires more than 10 GB of disk space

undir_input_dir=inputs-undirected
dir_input_dir=inputs-directed

# Undirected inputs for CC, GC, MIS, and MST
mkdir -p $undir_input_dir

wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/2d-2e20.sym.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/amazon0601.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/as-skitter.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/citationCiteseer.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/cit-Patents.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/coPapersDBLP.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/delaunay_n24.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/europe_osm.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/in-2004.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/internet.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/kron_g500-logn21.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/r4-2e23.sym.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/rmat16.sym.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/rmat22.sym.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/soc-LiveJournal1.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/USA-road-d.NY.egr
wget -nc -P $undir_input_dir --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/USA-road-d.USA.egr

# Directed inputs for SCC
g++ scripts/mm2ecl_real.cpp -I./library -o scripts/mm2ecl_real # Compile mm2ecl converter
mm2ecl=../scripts/mm2ecl_real

mkdir -p $dir_input_dir
cd $dir_input_dir

wget --no-check-certificate https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz
tar -xvzf circuit5M.tar.gz
rm circuit5M.tar.gz
${mm2ecl} circuit5M/circuit5M.mtx circuit5M.egr
rm -r circuit5M

wget --no-check-certificate https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage14.tar.gz
tar -xvzf cage14.tar.gz
rm cage14.tar.gz
${mm2ecl} cage14/cage14.mtx cage14.egr
rm -r cage14

wget --no-check-certificate https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20061104.tar.gz
tar -xvzf wikipedia-20061104.tar.gz
rm wikipedia-20061104.tar.gz
${mm2ecl} wikipedia-20061104/wikipedia-20061104.mtx wikipedia-20061104.egr
rm -r wikipedia-20061104

wget --no-check-certificate https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz
tar -xvzf web-Google.tar.gz
rm web-Google.tar.gz
${mm2ecl} web-Google/web-Google.mtx web-Google.egr
rm -r web-Google

wget --no-check-certificate https://suitesparse-collection-website.herokuapp.com/MM/Gleich/flickr.tar.gz
tar -xvzf flickr.tar.gz
rm flickr.tar.gz
${mm2ecl} flickr/flickr.mtx flickr.egr
rm -r flickr

wget -nc --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/inputs/large-meshes/klein-bottle.mesh-M-4-idx-0.egr
wget -nc --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/inputs/cold-flow.mesh.egr
wget -nc --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/inputs/large-meshes/toroid-hex.mesh-M-4-idx-0.egr
wget -nc --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/inputs/toroid-wedge.mesh.egr
wget -nc --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/inputs/star.mesh.egr
