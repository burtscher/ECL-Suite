#!/usr/bin/python3 -u

'''
This file is part of the ECL Suite version 1.0.

BSD 3-Clause License

Copyright (c) 2024, Brandon Alexander Burtchell and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/ECL-Suite.

Publication: This work is described in detail in the following paper.
Yiqian Liu, Avery VanAusdal, and Martin Burtscher. Performance Impact of Removing Data Races from GPU Programs. Proceedings of the IEEE International Symposium on Workload Characterization. September 2024.
'''


import os
import sys
import subprocess

exe_name = "minibench"
space = " "

# compute the compile command
def compile_cmd(lib_path, code_file_name):
    compiler = 'nvcc'
    optimize_flag = "-O3"
    #parallel_flag = "-arch=sm_" + arch_number
    parallel_flag = "-arch=native"
    library = "-I" + lib_path
    out_name = "-o" + space + exe_name
    return compiler + space + optimize_flag + space + parallel_flag + space + library + space + out_name + space + code_file_name

# compute the run command
def run_cmd(input):
    return "./" + exe_name + space + input

def compile_code(code_file, code_counter, num_codes, command):
    sys.stdout.flush()
    print("compile %s, %s out of %s programs\n" % (code_file, code_counter, num_codes))
    sys.stdout.flush()
    os.system(command)

def run_code(code_file, input_files, num_inputs, nput_path, num_runs):
    input_counter = 0
    for input_file in input_files:
        input_counter += 1
        print("running %s, %s out of %s inputs\n" % (code_file, input_counter, num_inputs))
        for i in range(num_runs):
            print("run %s out of %s\n" % (i + 1, num_runs))
            run = run_cmd(os.path.join(input_path, input_file)).split()
            sys.stdout.flush()
            subprocess.run(run)
            sys.stdout.flush()


def delete_exe(exe_name):
    if os.path.isfile(exe_name):
        os.system("rm " + exe_name)
    else:
        sys.exit('Error: compile failed')

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) != 5):
        sys.exit('USAGE: code_dir input_dir lib_path num_runs\n')

    # read inputs
    code_path = args_val[1]
    input_path = args_val[2]
    lib_path = args_val[3]
    num_runs = int(args_val[4])

    # list code files
    code_files = [f for f in os.listdir(code_path) if os.path.isfile(os.path.join(code_path, f))]
    num_codes = len(code_files)

    # list input files
    input_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    num_inputs = len(input_files)
    print("num_codes: %d\nnum_inputs: %d\n" % (num_codes, num_inputs))
    print("code_path: %s\ninput_path: %s\n" % (code_path, input_path))

    # compile and run the codes
    code_counter = 0
    model = '.cu'
    for code_file in code_files:
        if code_file.endswith(model):
            code_counter += 1
            compile_code(code_file, code_counter, num_codes, compile_cmd(lib_path, os.path.join(code_path, code_file)))
            run_code(code_file, input_files, num_inputs, input_path, num_runs)
            delete_exe(exe_name)
        else:
            sys.exit('File %s does not match the programming model %s.' % (code_file, model))
