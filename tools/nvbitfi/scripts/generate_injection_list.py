# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys, re, string, os, operator, math, datetime, random
import params as p
import common_functions as cf

MAX_INJ = p.NUM_INJECTIONS
verbose = False
inj_mode = ""


#################################################################
# Generate injection list of each
#   - app
#   - instruction group
#   - bit-flip model
#################################################################
def write_injection_list_file(
    app, inj_mode, igid, bfm, num_injections, total_count, countList
):
    if verbose:
        print("total_count = %d, num_injections = %d" % (total_count, num_injections))
    fName = (
        p.app_log_dir[app]
        + "/injection-list/mode"
        + inj_mode
        + "-igid"
        + str(igid)
        + ".bfm"
        + str(bfm)
        + "."
        + str(num_injections)
        + ".txt"
    )
    print(fName)
    f = open(fName, "w")

    while num_injections > 0 and total_count != 0:  # first two are kname and kcount
        num_injections -= 1
        injection_num = random.randint(
            0, total_count
        )  # randomly select an injection index
        if igid == "rf":
            [inj_kname, inj_kcount, inj_icount] = cf.get_rf_injection_site_info(
                countList, injection_num, True
            )  # convert injection index to [kname, kernel count, inst index]
            inj_op_id_seed = (
                p.num_regs[app][inj_kname] * random.random()
            )  # register selection
        else:
            [inj_kname, inj_kcount, inj_icount] = cf.get_injection_site_info(
                countList, injection_num, igid
            )  # convert injection index to [kname, kernel count, inst index]
            inj_op_id_seed = random.random()
        inj_bid_seed = random.random()
        # print("inj_icount = %d" % inj_icount)
        selected_str = (
            inj_kname
            + " "
            + str(inj_kcount)
            + " "
            + str(inj_icount)
            + " "
            + str(inj_op_id_seed)
            + " "
            + str(inj_bid_seed)
            + " "
        )
        if verbose:
            print("%d/%d: Selected: %s" % (num_injections, total_count, selected_str))
        f.write(selected_str + "\n")  # print injection site information
    f.close()


#################################################################
# Generate injection list of each app for
# (1) RF AVF (for each error model)
# (2) instruction-level value injections (for each error model and instruction type)
# (3) instruction-level address injections (for each error model and instruction type)
#################################################################
def gen_lists(app, countList, inj_mode):
    if inj_mode == p.RF_MODE:  # RF injection list
        total_count = (
            cf.get_total_insts(countList, True)
            if inj_mode == p.RF_MODE
            else cf.get_total_insts(countList, False)
        )
        for bfm in p.rf_bfm_list:
            write_injection_list_file(
                app, inj_mode, "rf", bfm, MAX_INJ, total_count, countList
            )
    elif inj_mode == p.INST_VALUE_MODE:  # instruction value injections
        total_icounts = cf.get_total_counts(countList)
        for igid in p.inst_value_igid_bfm_map:
            for bfm in p.inst_value_igid_bfm_map[igid]:
                write_injection_list_file(
                    app,
                    inj_mode,
                    igid,
                    bfm,
                    MAX_INJ,
                    total_icounts[igid - p.NUM_INST_GROUPS],
                    countList,
                )
    elif inj_mode == p.INST_ADDRESS_MODE:  # instruction value injections
        total_icounts = cf.get_total_counts(countList)
        for igid in p.inst_address_igid_bfm_map:
            for bfm in p.inst_address_igid_bfm_map[igid]:
                write_injection_list_file(
                    app,
                    inj_mode,
                    igid,
                    bfm,
                    MAX_INJ,
                    total_icounts[igid - p.NUM_INST_GROUPS],
                    countList,
                )


igid_to_inst_type = [[0], [1], [2], [3], [4], [5], [0, 1, 2, 3, 5], [0, 1, 2, 5]]


def gen_list_for_test_static_insts(app, inj_mode, filtered_id_map):
    filtered_id_map_by_second_elem = {}
    for entry in filtered_id_map:
        key = entry[1]
        if key not in filtered_id_map_by_second_elem:
            filtered_id_map_by_second_elem[key] = []
        filtered_id_map_by_second_elem[key].append(entry)

    if inj_mode == p.INST_VALUE_MODE:
        for igid in p.inst_value_igid_bfm_map:
            for bfm in p.inst_value_igid_bfm_map[igid]:
                fName = (
                    p.app_log_dir[app]
                    + "/injection-list/mode"
                    + inj_mode
                    + "-igid"
                    + str(igid)
                    + ".bfm"
                    + str(bfm)
                    + ".txt"
                )
                f = open(fName, "w")
                for key, entries in filtered_id_map_by_second_elem.items():
                    filtered_entries = [
                        entry
                        for entry in entries
                        if entry[3] in igid_to_inst_type[igid] and entry[4] != 0
                    ]

                    base = len(filtered_entries) // p.inst_devide_total_num
                    remainder = len(filtered_entries) % p.inst_devide_total_num

                    # 计算当前分片的起始和结束位置
                    if p.inst_devide_id < remainder:
                        start = p.inst_devide_id * (base + 1)
                        end = start + (base + 1)
                    else:
                        start = (
                            remainder * (base + 1)
                            + (p.inst_devide_id - remainder) * base
                        )
                        end = start + base

                    # 生成索引列表（Python的range是左闭右开区间）
                    sampled_indices = list(range(start, end))

                    # 打印结果
                    print(
                        f"Sampled indices for partition {p.inst_devide_id} of {p.inst_devide_total_num}:"
                    )
                    print(sampled_indices)
                    print(len(filtered_entries))
                    product_sum = 0
                    for sublist in filtered_entries:
                        product_sum += sublist[4]
                    result = (
                        product_sum / len(filtered_entries) if filtered_entries else 0
                    )
                    print("thread_per_inst", result)
                    sample_sum = 0
                    for i, entry in enumerate(filtered_entries):
                        if i in sampled_indices:
                            injection_thread_num = 384
                            sample_sum += injection_thread_num
                            injection_threads = random.choices(
                                range(entry[4]), k=int(injection_thread_num)
                            )
                            for thread in injection_threads:
                                inj_op_id_seed = random.random()
                                inj_bid_seed = random.random()
                                f.write(
                                    entry[0]
                                    + " "
                                    + str(entry[1])
                                    + " "
                                    + str(thread)
                                    + " "
                                    + str(inj_op_id_seed)
                                    + " "
                                    + str(inj_bid_seed)
                                    + " "
                                    + str(entry[2])
                                    + "\n"
                                )
                    print("sample_sum", sample_sum)

                f.close()


#################################################################
# Starting point of the script
#################################################################
def main():
    # if len(sys.argv) == 2:
    # 	inj_mode = sys.argv[1] # rf or inst_value or inst_address
    # else:
    # 	print ("Usage: ./script-name <rf or inst>")
    # 	print ("Only one mode is currently supported: inst_value")
    # 	exit(1)
    inj_mode = "inst_value"  # we only support inst_value mode in NVBitFI as of now (March 25, 2020)

    # actual code that generates list per app is here
    for app in p.apps:
        print("\nCreating list for %s ... " % (app))
        os.system(
            "mkdir -p %s/injection-list" % p.app_log_dir[app]
        )  # create directory to store injection list

        countList = cf.read_inst_counts(p.app_log_dir[app], app)
        total_count = (
            cf.get_total_insts(countList, True)
            if inj_mode == p.RF_MODE
            else cf.get_total_insts(countList, False)
        )
        if total_count == 0:
            print("Something is not right. Total instruction count = 0\n")
            sys.exit(-1)

        # gen_lists(app, countList, inj_mode)
        ##################################################################
        # Begin of the test static inst code
        ##################################################################

        id_map = cf.read_id_map(p.app_log_dir[app], app)
        filtered_id_map = [entry for entry in id_map if entry[3] not in [100, 3, 4]]
        gen_list_for_test_static_insts(app, inj_mode, filtered_id_map)

        print("Output: Check %s" % (p.app_log_dir[app] + "/injection-list/"))


if __name__ == "__main__":
    main()
