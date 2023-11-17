# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument
"""Tools to compare libraries."""
from typing import List, Tuple, Iterable, Union

import tvm
import tvm.testing


class LibCompareVMInstrument:
    """Instrument class to compare libs.

    This class build an instrument function that
    pair tests an existing compiled relax vm implementation
    and an extra module, which can sit in another backend
    but offer a same subset of compiled TIR functions.

    The instrumentation enables us to automatically
    check and compare each ops being called in the pipeline
    by looking up the same name in the provided mod and run testing.

    Parameters
    ----------
    mod: runtime.Module
        The module of interest to be validated.

    device: runtime.Device
        The device to run the target module on.

    verbose: bool
        Whether print out messages.

    rtol: float
        rtol used in validation

    atol: float
        atol used in validation
    """

    def __init__(self, mod, device, verbose=True, rtol=1e-5, atol=1e-5):
        self.mod = mod
        self.device = device
        self.verbose = verbose
        self.counter = 0
        self.rtol = rtol
        self.atol = atol

    def compare(
        self,
        name: str,
        ref_args: Union[List[tvm.nd.NDArray], Tuple[tvm.nd.NDArray, ...]],
        new_args: Union[List[tvm.nd.NDArray], Tuple[tvm.nd.NDArray, ...]],
        ret_indices: Iterable[int],
    ):
        """Comparison function, can be overloaded.

        Parameters
        ----------
        name: str
            Name of the function.

        ref_args:
            The reference arguments.

        new_args:
            The args to be passed to the comparison function.

        ret_indices:
            List of indices to validate return values.
        """
        my_func = self.mod.get_function(name, query_imports=True)
        if self.verbose:
            print(f"[{self.counter}] Validating {name} ...")
        my_func(*new_args)
        for rindex in ret_indices:
            tvm.testing.assert_allclose(
                new_args[rindex].numpy(), ref_args[rindex].numpy(), atol=self.atol, rtol=self.rtol
            )
        if self.verbose:
            for i in range(len(new_args)):
                print("The {}th argument value of {}:\n{}".format(i, name, new_args))
            print(f"[{self.counter}] Validating {name}, passed.")
        self.counter += 1

    def skip_instrument(self, func, name, before_run, ret_val, *args):
        return False

    def __call__(self, func, name, before_run, ret_val, *args):
        if before_run:
            return
        if name.startswith("vm.builtin."):
            return
        if any(not isinstance(x, tvm.nd.NDArray) for x in args):
            return
        try:
            self.mod.get_function(name, query_imports=True)
        except AttributeError:
            if self.verbose:
                print(f"Cannot find {name}, skip...")
            return

        if self.skip_instrument(func, name, before_run, ret_val, *args):
            return

        new_args = []
        # not always true, true for most ops.
        ret_indices = (len(args) - 1,)
        temp_args = []
        for i, arg in enumerate(args):
            arr = tvm.nd.empty(arg.shape, arg.dtype, device=self.device)
            # copy from cpu since we look at different device
            if i not in ret_indices:
                temp_cpu = arg.copyto(tvm.cpu())
                temp_args.append(temp_cpu)
                arr.copyfrom(temp_cpu)
            new_args.append(arr)
        # wait until all copy complete before we release temp_cpu
        self.device.sync()
        self.compare(name, args, new_args, ret_indices)


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod, device, time_eval, skip_rounds=0):
        super().__init__(mod, device, True)
        self.time_eval = time_eval
        self.time_eval_results = {}
        self.visited = set([])
        self.skip_rounds = skip_rounds
        self.atol = 1e-2
        self.rtol = 1e-3

    def skip_instrument(self, func, name, before_run, ret_val, *args):
        print(f"run {name}")
        if name.startswith("shape_func"):
            return True
        if self.counter < self.skip_rounds:
            self.counter += 1
            print(f"[{self.counter}] Skip validating {name}..")
            return True
        if name in self.visited:
            if self.time_eval and name in self.time_eval_results:
                record = self.time_eval_results[name]
                self.time_eval_results[name] = (record[0], record[1] + 1)
            return True
        self.visited.add(name)
        return False

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):
        super().compare(name, ref_args, new_args, ret_indices)

        if self.time_eval and name not in self.time_eval_results:
            res = self.mod.time_evaluator(
                name, self.device, number=20, repeat=3  # , cache_flush_bytes=256 * 10**6
            )(*new_args)
            self.time_eval_results[name] = (res.mean, 1)
            print(f"Time-eval result {name} on {self.device}: {res}")
