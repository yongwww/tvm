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
"""Global Info."""
import tvm
from typing import List
from tvm.ir import RelayExpr as Expr
from tvm.runtime.object import Object
from . import _ffi_api


class GlobalInfo(Object):
    """Base node for all global info that can appear in the IR"""

    def __eq__(self, other):
        """Compare two struct info for structural equivalence."""
        return tvm.ir.structural_equal(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Overload with structural equality."""
        return super().__eq__(other)


class ReturnGlobalInfo(GlobalInfo):
    """ReturnGlobalInfo in the IR.

    Parameters
    ----------
    return_exprs : List[Expr]
        The expressions to be returned.
    """

    return_exprs: List[Expr]

    def __init__(self, return_exprs: List[Expr]) -> None:
        print("yes entering ReturnGlobalInfo in global_info.py, return_exprs: ", return_exprs)
        self.__init_handle_by_constructor__(_ffi_api.ReturnGlobalInfo, return_exprs)

    def add():
        pass

    def update(return_exprs: List[Expr]):
        pass

    def get() -> GlobalInfo:
        pass

    def get_exprs(self):
        return self.return_exprs


class DummyGlobalInfo(GlobalInfo):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DummyGlobalInfo,
        )
