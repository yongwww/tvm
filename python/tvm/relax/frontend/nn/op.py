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
# pylint: disable=too-many-lines,invalid-name,protected-access,redefined-outer-name
"""nn.Tensor operators."""
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tvm import tir as _tir, relax as _rx

from ... import expr as rx
from ... import op as _op
from ...block_builder import BlockBuilder
from ...struct_info import TensorStructInfo, TupleStructInfo
from .core import Tensor, get_default_dtype
from .spec import SpecBuilder

IntExpr = Union[int, _tir.PrimExpr]


def _wrap_nested(expr: rx.Expr, name: str) -> Union[Tensor, Sequence[Tensor]]:
    """Wrap the given relax.Expr, emit it using the current BlockBuilder,
    and automatically handle nested cases if the expr represents a Tuple.

    Parameters
    ----------
    expr : relax.Expr
        The Expr to be wrapped.

    name : str
        Name hint.

    Returns
    -------
    result : Union[Tensor, Tuple[Tensor]]
        The computed result.
    """
    if not isinstance(expr, rx.DataflowVar):
        expr = BlockBuilder.current().emit(expr, name)
    if isinstance(expr.struct_info_, TensorStructInfo):
        return Tensor(_expr=expr)
    if isinstance(expr.struct_info_, TupleStructInfo):
        return tuple(
            _wrap_nested(
                rx.TupleGetItem(expr, i),
                name=f"{name}.{i}",
            )
            for i in range(len(expr.struct_info_.fields))
        )
    raise TypeError(f"Unsupported return type: {expr.struct_info_}")


def unsqueeze(x: Tensor, dim: int, name: str = "unsqueeze") -> Tensor:
    """Add a new axis to a tensor

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    dim : int
        Dimension to expand.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Expanded result.
    """
    return _wrap_nested(_op.expand_dims(x._expr, dim), name)


def concat(x: List[Tensor], dim: int, name: str = "concat") -> Tensor:
    """Concatenate a list of tensors along an axis.

    Parameters
    ----------
    x : List[Tensor]
        List of tensors to concatenate.
    dim : int
        Dimension to concatenate upon.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Expanded result.
    """
    # Convert tensors to expressions.
    x = [t._expr for t in x]
    return _wrap_nested(_op.concat(x, dim), name)

    # def stack(x: List[Tensor], dim: int, name: str = "concat") -> Tensor:
    """Concatenate a list of tensors along an axis.

    Parameters
    ----------
    x : List[Tensor]
        List of tensors to concatenate.
    dim : int
        Dimension to concatenate upon.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Expanded result.
    """
    # Convert tensors to expressions.
    ### todo (*
    # x = [t._expr for t in x]
    # return _wrap_nested(_op.concat(x, dim), name)


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = add(a, b)
    """
    return _wrap_nested(_op.add(a._expr, b._expr), name)


def sin(x: Tensor, name: str = "sin") -> Tensor:
    """Compute element-wise sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.sin(x._expr), name)


def cos(x: Tensor, name: str = "cos") -> Tensor:
    """Compute element-wise cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.cos(x._expr), name)


def subtract(a: Tensor, b: Tensor, name: str = "subtract") -> Tensor:
    """Subtraction with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = subtract(a, b)
    """
    return _wrap_nested(_op.subtract(a._expr, b._expr), name)


def multiply(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = multiply(a, b)
    """
    return _wrap_nested(_op.multiply(a._expr, b._expr), name)


def divide(a: Tensor, b: Tensor, name: str = "divide") -> Tensor:
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = divide(a, b)
    """
    return _wrap_nested(_op.divide(a._expr, b._expr), name)


def cumsum(x: Tensor, axis: Optional[int] = None, dtype=None, name: str = "cumsum") -> Tensor:
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : Optional[Union[str, DataType]]
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    """
    return _wrap_nested(_op.cumsum(x._expr, axis, dtype), name)


def chunk(x: Tensor, chunks: int, dim: int = 0, name: str = "chunk") -> Tensor:
    """Split a tensor along dim into the specified number of chunks.

    Parameters
    ----------
    x : Tensor
        Input tensor to be split.
    chunks : int
        Number of pieces to slice x into.
    dim : int
        Which dimension to split x.
    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tuple[Tensor]
        A tuple with chunks elements containing slices of x.
    """
    return _wrap_nested(_op.split(x._expr, chunks, dim), name)


def strided_slice(
    x: Tensor,
    axes: List[int],
    begin: List[int],
    end: List[int],
    strides: Optional[List[int]] = None,
    assume_inbound: bool = False,
    name: str = "strided_slice",
) -> Tensor:
    """Strided slice of a tensor.

    Parameters
    ----------
    x : Tensor
        The source tensor to be sliced.

    axes : List[int]
        Axes along which slicing is applied.

    begin : List[PrimExprLike]
        The indices to begin with in the slicing, inclusive.

    end : List[PrimExprLike]
        The indices indicating end of the slice, exclusive.

    strides : Optional[List[PrimExprLike]]
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.
        If not specified, it by default is an list of ones of the same length as `axes`.

    assume_inbound : bool
        Whether to assume the indices are in bound. If it is set to false,
        out of bound indices will be clipped to the bound.

    name : str
        Name hint for this operation.

    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    strided_slice require the input `begin`, `end` and `strides` to have the
    same length as `axes`.
    """

    if isinstance(begin, Tensor):
        begin = begin._expr
    if isinstance(end, Tensor):
        end = end._expr
    return _wrap_nested(_op.strided_slice(x._expr, axes, begin, end, strides, assume_inbound), name)  # type: ignore


def dynamic_strided_slice(
    x: Tensor,
    begin: List[int],
    end: List[int],
    strides: Optional[List[int]] = None,
    name: str = "dynamic_strided_slice",
) -> Tensor:
    """Dynamic strided slice of a tensor. `begin`, `end`, `strides` can be computed at runtime.

    Parameters
    ----------
    x : Expr
        The source tensor to be sliced.

    begin : Expr
        The indices to begin with in the slicing, inclusive.

    end : Expr
        The indices indicating end of the slice, exclusive.

    strides : Expr
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.
        If not specified, it by default is an list of ones of the same length as `axes`.

    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    dyn_strided_slice require the input `begin`, `end` and `strides` to have the
    same length as rank of `data` tensor.
    """

    return _wrap_nested(_op.dynamic_strided_slice(x._expr, begin._expr, end._expr, strides._expr), name)  # type: ignore


def einsum(operands, subscripts, name: str = "einsum"):
    """Evaluates the Einstein summation convention on data

    Parameters
    ----------
    operands : Union(List[relax.Expr], Tuple[relax.Expr])
        A list of expression.

    subscripts : str
        The einsum expression string.

    name : str
        Name hint for this operation.


    Returns
    -------
    result : relax.Expr
        The output from the einsum op.
    """
    operands = [operand._expr if isinstance(operand, Tensor) else operand for operand in operands]
    return _wrap_nested(_op.einsum(operands, subscripts), name)  # type: ignore


def equal(x1: Tensor, x2: Tensor, name: str = "equal") -> Tensor:
    """Broadcasted element-wise test for (lhs == rhs).

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return _wrap_nested(_op.equal(x1._expr, x2._expr), name)  # type: ignore


def not_equal(x1: Tensor, x2: Tensor, name: str = "equal") -> Tensor:
    """Broadcasted element-wise test for (lhs != rhs).

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return _wrap_nested(_op.not_equal(x1._expr, x2._expr), name)  # type: ignore


def sum(
    x: Tensor,
    axis: Optional[Union[int, List[int]]] = None,
    keepdims: bool = False,
    name: str = "sum",
) -> Tensor:
    """Computes the sum of tensor elements over given axes.

    Parameters
    ----------
    x : Tensor
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _wrap_nested(_op.sum(x._expr, axis, keepdims), name)  # type: ignore


def expand_dims(x: Tensor, axis: Union[int, List[int]], name: str = "expand_dims") -> Tensor:
    """Insert new axes at the positions given by `axis`.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axis : Union[int, List[int]]
        The axes at which the input array are expanded.
        All values are required to lie in range `[-data.ndim - 1, data.ndim]`, with the convention
        of negative indexing.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : relax.Expr
        The transformed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _wrap_nested(_op.expand_dims(x._expr, axis), name)  # type: ignore


def flatten(data, start_dim=0, end_dim=-1):
    start = start_dim
    end = end_dim
    dshape = data.shape
    ndim = len(dshape)
    if start < 0:
        start += ndim
    if end < 0:
        end += ndim
    assert start <= end, "start dim cannot come after end dim"
    new_shape = [0] * start

    new_shape.append(-1)
    squeeze_axes = []
    for i in range(start + 1, end + 1):
        new_shape.append(1)
        squeeze_axes.append(i)
    for _ in range(end + 1, ndim):
        new_shape.append(0)
    out = reshape(data, new_shape)
    if squeeze_axes:
        out = squeeze(out, axis=squeeze_axes)
    return out


def matmul(a: Tensor, b: Tensor, out_dtype: Optional[str] = None, name: str = "matmul") -> Tensor:
    """General matrix multiplication of two tensors, with broadcasting on batched dimensions.

    The semantics and output shape deduction rule is specified as
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    out_dtype: Optional[Union[str, DataType]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the same as input dtype.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = matmul(a, b)
    """
    return _wrap_nested(_op.matmul(a._expr, b._expr, out_dtype=out_dtype), name)


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple]] = 1,
    padding: Optional[Union[int, Tuple, str]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    name: str = "conv1d",
) -> Tensor:
    r"""1D convolution.

    This operator takes the weight as the 1D convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCW`
    and kernel_layout is `OIW`, conv1d takes in
    a data Tensor with shape `(batch_size, in_channels, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_w)`,
    where `kernel_w` is the length of the `W` kernel dimension,
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, x] = \sum_{dx, k}
           \mbox{data}[b, k, \mbox{strides} * x + dx] *
           \mbox{weight}[c, k, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCW` for data and `OIW` for weight), perform the computation,
    then convert to the out_layout.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    weight : Tensor
        The weight expressions.

    bias : Optional[Tensor]
        Optional bias tensor of shape [O].

    strides : Optional[Union[int, Tuple]]
        The strides of convolution. It is required to have length 1.

    padding : Optional[Union[int, Tuple, str]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1 or 2.

    dilation : Optional[Union[int, Tuple]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length 1.

    groups : Optional[int]
        Number of groups to split the input into for grouped convolution.
        The number of input and output channels should be divisible by the number of groups.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    conv_out = _op.nn.conv1d(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1]))

    return _wrap_nested(conv_out, name)


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple]] = 1,
    padding: Optional[Union[int, Tuple, str]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    name: str = "conv2d",
) -> Tensor:
    """Applies a 2D convolution over an input image composed of sevaral input planes

    Parameters
    ----------
    x : Tensor
        Input tensor of shape [B, N, H, W]

    weight : Tensor
        Filters of shape [O, N/groups, kH, kW]

    bias : Optional[Tensor]
        Optional bias tensor of shape [O].

    stride : Optional[Union[int, Tuple]]
        The stride of the convolving kernel. Can be a single number
        or tuple of (sH, sW).

    padding : Optional[[Union[int, Tuple]]]
        Implicit paddings on both sides of the input.

    dilation : Optional[Union[int, Tuple]]
        The spacing between kernel elements. Can be a single number of tuple (dH, dW).

    groups : Optional[int]
        Split input into a number of groups.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result with shape [B, O, oH, oW].
    """
    conv_out = _op.nn.conv2d(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1, 1]))

    return _wrap_nested(conv_out, name)


def conv1d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple[int]]] = 1,
    padding: Optional[Union[int, Tuple[int, ...]]] = 0,
    output_padding: Optional[Union[int, Tuple[int]]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    name: str = "conv1d_transpose",
) -> Tensor:
    """1D transposed convolution operator.

    This operator can be seen as the gradient operator of conv1d.

    The output shape can be explained in the simple case when `data_layout == "NCW"` and
    `kernel_layout == "IOW"`. Suppose `data` has shape `(N, in_channel, in_w)`, `weight` has
    shape `(in_channel, out_channel, weight_w)`, we need to assure that `in_channel % groups == 0`.
    The shape of the output will be `(N, out_channel * groups, out_w)`, where

    - `out_w = ((in_w - 1) * strides[0] + weight_w - 2 * padding[0] + output_padding[0])`

    Parameters
    ----------
    data : Tensor
        The input data to the operator.

    weight : Tensor
        The weight tensor.

    strides : Union[int, Tuple[int]]
        The strides of convolution. It is required to have length 1.

    padding : Union[int, Tuple[int, ...]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1 or 2.

    output_padding : Union[int, Tuple[int, ...]], optional
        Used to disambiguate the output shape.

    dilation : Union[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1.

    groups : int
        Number of groups to split the input into for grouped convolution.
        The number of input and output channels should be divisible by the number of groups.

    data_layout : str
        Layout of the input.

    kernel_layout : str
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    out_dtype : Optional[Union[str, DataType]]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    conv_out = _op.nn.conv1d_transpose(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1]))

    return _wrap_nested(conv_out, name)


def conv2d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    output_padding: Union[int, Tuple[int, int]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: Optional[int] = 1,
    name: str = "conv2d_transpose",
) -> Tensor:
    """2D transposed convolution operator.

    This operator is intended to be the gradient operator of conv2d. That means, if

    `out = conv2d(data, weight, strides, padding, dilation)`,

    The gradient w.r.t. data can be calculated as follows:

    `data_grad = conv2d_transpose(out_grad, weight, strides, padding, output_padding, dilation)`,

    where `output_padding` is a parameter used to determine the output shape.

    The output shape can be explained in the simple case when `data_layout == "NCHW"` and
    `kernel_layout == "IOHW"`. Suppose `data` has shape `(N, in_channel, in_h, in_w)`, `weight` has
    shape `(in_channel, out_channel, weight_h, weight_w)`, we need to assure that
    `in_channel % groups == 0`. The shape of the output will be
    `(N, out_channel * groups, out_h, out_w)`, where

    - `out_h = ((in_h - 1) * strides[0] + weight_h - 2 * padding[0] + output_padding[0])`
    - `out_w = ((in_w - 1) * strides[1] + weight_w - 2 * padding[1] + output_padding[1])`

    Parameters
    ----------
    data : Tensor
        The input data to the operator.

    weight : Tensor
        The weight tensor.

    strides : Union[int, Tuple[int, int]]
        The strides of convolution. It is required to have length either 1 or 2.

    padding : Union[int, Tuple[int, ...]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1, 2 or 4.

    output_padding : Union[int, Tuple[int, ...]], optional
        Used to disambiguate the output shape.

    dilation : Union[int, Tuple[int, int]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1 or 2.

    groups : int
        Number of groups to split the input into for grouped convolution.
        The number of input and output channels should be divisible by the number of groups.

    data_layout : str
        Layout of the input.

    kernel_layout : str
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    out_dtype : Optional[Union[str, DataType]]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    conv_out = _op.nn.conv2d_transpose(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        print("TODO (yongwww): add test case to cover this bias", bias)
        # conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1]))

    return _wrap_nested(conv_out, name)


def maximum(x1: Tensor, x2: Tensor, name: str = "maximum"):
    """Element-wise maximum

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = maximum(a, b)
    """
    return _wrap_nested(_op.maximum(x1._expr, x2._expr), name)


def minimum(x1: Tensor, x2: Tensor, name: str = "minimum"):
    """Element-wise minimum

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = minimum(a, b)
    """
    return _wrap_nested(_op.minimum(x1._expr, x2._expr), name)


def mean(x: Tensor, axis=None, keepdim: bool = False, name: str = "mean"):
    """Computes the mean of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input tensor.
        Negative indexing is supported.

    keepdim : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    """
    return _wrap_nested(_op.mean(x._expr, axis, keepdim), name)


def pow(x1: Tensor, x2: Tensor, name: str = "pow"):
    """Power with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    """
    return _wrap_nested(_op.power(x1._expr, x2._expr), name)


def sqrt(x: Tensor, name: str = "sqrt"):
    """Compute element-wise square root of the input data.

    Parameters
    ----------
    x : Tensor
        The input data

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    """
    return _wrap_nested(_op.sqrt(x._expr), name)


def broadcast_to(x: Tensor, shape: Sequence[IntExpr], name: str = "broadcast_to") -> Tensor:
    """Broadcasts a tensor to a specified shape.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    shape : Sequence[IntExpr]
        The target shape.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The broadcasted tensor.
    """
    return _wrap_nested(_op.broadcast_to(x._expr, shape), name)


def permute_dims(x: Tensor, axes: Optional[List[int]] = None, name: str = "permute_dims") -> Tensor:
    """Permutes the dimensions of an array.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order, reverse order if not specified.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The permuted result.
    """
    return _wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


def transpose(x: Tensor, dim0: int, dim1: int, name: str = "transpose") -> Tensor:
    """transpose the array.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    dim0 : int
        The first dimension to be transposed.

    dim1 : int
        The second dimension to be transposed

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The transposed result.
    """
    # TODO (yongwww): merge this into permute_dims as a special case
    ndim = len(x.shape)
    axes = data = list(range(ndim))
    data[dim0] = dim1
    data[dim1] = dim0
    return _wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


def reshape(x: Tensor, shape: Sequence[IntExpr], name="reshape") -> Tensor:
    """Reshape the input array.

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            x.shape = (2, 3, 4), shape = (6, 1, -1), result.shape = (6, 1, 4)
            x.shape = (2, 3, 4), shape = (3, -1, 8), result.shape = (3, 1, 8)
            x.shape = (2, 3, 4), shape = (-1,), result.shape = (24,)

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    shape : Sequence[IntExpr]
        The new shape. Should be compatible with the original shape.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The reshaped result.

    Note
    ----
    The ``-1`` inference is only performed at compile-time.
    That is to say, in any case the dimension length of ``-1`` cannot be inferred in
    compile-time, an error will be thrown.
    """
    return _wrap_nested(_op.reshape(x._expr, shape), name)


def expand(x: Tensor, sizes: Sequence[IntExpr], name="expand") -> Tensor:
    """Expand the input array.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    sizes : Sequence[IntExpr]
        The new shape. Should be compatible with the original shape.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The reshaped result.

    """

    shape = x.shape
    ndims = len(shape)
    out = x._expr

    out_dims = len(sizes)
    if ndims < out_dims:
        num_newaxis = out_dims - ndims
        out = _op.expand_dims(out, axis=0, num_newaxis=num_newaxis)
        shape = [1] * num_newaxis + shape

    for i in range(out_dims):
        if sizes[i] != -1 and shape[i] == 1:
            out = _op.repeat(out, sizes[i], axis=i)

    return _wrap_nested(out, name)


def repeat(x: Tensor, repeats: int, axis: Optional[int] = None, name="repeat") -> Tensor:
    """Repeats elements of an array.

    Parameters
    ----------
    data : Tensor
        The input tensor.

    repeats : int
        The number of repetitions.

    axis: Optional[int]
        The axis along which to repeat values. The negative numbers are interpreted
        counting from the backward. By default, use the flattened input array, and
        return a flat output array.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The computed result.

    Examples
    --------
    .. code-block:: python

        np_x = numpy.array([[1, 2], [3, 4]])
        x = Tensor.from_const(np_x)
        lv1 = repeat(x, repeats=2) # lv1 == [1, 1, 2, 2, 3, 3, 4, 4]
        lv2 = repeat(x, repeats=2, axis=1)   # lv2 == [[1., 1., 2., 2.],
                                             #         [3., 3., 4., 4.]]
    """
    return _wrap_nested(_op.repeat(x._expr, repeats, axis), name)


def tile(data: Tensor, repeats: Union[int, Tuple[int], List[int]], name="tile") -> Tensor:
    """Construct an array by repeating data the number of times given by repeats.

    If repeats has length l, and data has dimension d, the result will have dimension of max(l, d).

    If d < l, data is promoted to be l-dimensional by prepending new axes. So a shape (3,) Tensor is
    promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not
    the desired behavior, promote data to d-dimensions manually before calling this function.

    If d > l, reps is promoted to length d by pre-pending 1's to it. Thus for a data of shape
    (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).

    Parameters
    ----------
    data : Tensor
        The input data to the operator.

    repeats : Union[int, Tuple[int], List[int]]
        The number of repetitions of data along each axis.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = R.const([[1, 2], [3, 4]])
        lv1 = R.tile(x, reps=(2, 3)) # lv1 = [[1., 2., 1., 2., 1., 2.],
                                     #        [3., 4., 3., 4., 3., 4.],
                                     #        [1., 2., 1., 2., 1., 2.],
                                     #        [3., 4., 3., 4., 3., 4.]]
        lv2 = R.tile(x, reps=2) # lv2 = [[1., 2., 1., 2.],
                                #        [3., 4., 3., 4.]]
    """
    if isinstance(repeats, int):
        repeats = [repeats]

    return _wrap_nested(_op.tile(data._expr, repeats), name)


_op.sum


def squeeze(x: Tensor, axis: int = -1, name: str = "squeeze") -> Tensor:
    """Squeeze axes in the array.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    axis : Optional[Union[int, List[int]]
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The squeezed result.
    """
    return _wrap_nested(_op.squeeze(x._expr, axis), name)


def take(x: Tensor, indices: Tensor, axis: Optional[int] = None, name="take") -> Tensor:
    """Take elements from a tensor along an axis.
    Its semantic is mostly similar to `numpy.take`
    (https://numpy.org/doc/stable/reference/generated/numpy.take.html),
    which can cover `torch.take` (https://pytorch.org/docs/stable/generated/torch.take.html) and
    `onnx.gather` (https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13).

    Parameters
    ----------
    x : Tensor
        The source tensor.

    indices : Tensor
        The indices of the values to extract.

    axis : Optional[int]
        The axis over which to select values.
        If it is none, the input tensor is required to be one-dimensional.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The taken result.
    """
    return _wrap_nested(_op.take(x._expr, indices._expr, axis), name)


def where(condition: Tensor, x1: Tensor, x2: Tensor, name: str = "where") -> Tensor:
    """Selecting elements from either the input tensors depending on the value of the
    condition.

    For a given position, return the corresponding value in `x1` if `condition` is True,
    and return the corresponding value in `x2` otherwise.

    Parameters
    ----------
    condition : Tensor
        When True, yield `x1`; otherwise, yield `x2`.
        Must be broadcasting compatible with `x1` and `x2`.
        Must have boolean dtype.

    x1 : Tensor
        The first input tensor.
        Must be broadcasting compatible with `condition` and `x2`.

    x2 : Tensor
        The second input tensor.
        Must be broadcasting compatible with `condition` and `x1`.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The taken result.
    """
    return _wrap_nested(_op.where(condition._expr, x1._expr, x2._expr), name)


def If(condition: Tensor, x1: Tensor, x2: Tensor, name: str = "If") -> Tensor:
    """Selecting elements from either the input tensors depending on the value of the
    condition.

    For a given position, return the corresponding value in `x1` if `condition` is True,
    and return the corresponding value in `x2` otherwise.

    Parameters
    ----------
    condition : Tensor
        When True, yield `x1`; otherwise, yield `x2`.
        Must be broadcasting compatible with `x1` and `x2`.
        Must have boolean dtype.

    x1 : Tensor
        The first input tensor.
        Must be broadcasting compatible with `condition` and `x2`.

    x2 : Tensor
        The second input tensor.
        Must be broadcasting compatible with `condition` and `x1`.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The taken result.
    """
    if_stmt = rx.If(condition._expr, x1._expr, x2._expr)
    m = _tir.Var("m", dtype="int64")
    rx._update_struct_info(if_stmt, _rx.TensorStructInfo([1, 1, m, 256], "float32"))
    # if_stmt.struct_info =
    print("if_stmt sinifo: ", if_stmt.struct_info)
    print("if_stmt sinifo.shape: ", if_stmt.struct_info.shape)

    return _wrap_nested(if_stmt, name)


def astype(x: Tensor, dtype: str, name: str = "astype") -> Tensor:
    """Cast input tensor to the given data type.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    dtype: str
        The target data type

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The casted result.
    """
    # If trying to cast to same dtype as x, skip casting.
    if x.dtype == dtype:
        return x
    return _wrap_nested(_op.astype(x._expr, dtype), name)


def relu(x: Tensor, name: str = "relu") -> Tensor:
    """Rectified Linear Unit (ReLU) activation function.

    .. math::
        \text{ReLU}(x) = \text{max}(x, 0)

    Parameters
    ----------
    x : Tensor
        The input data.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return _wrap_nested(_op.nn.relu(x._expr), name)


def silu(x: Tensor, name: str = "silu") -> Tensor:
    r"""Sigmoid Linear Unit function

    .. math::
        \text{SiLU}(x) = x * \text{sigmoid}(x)

    Parameters
    ----------
    data : Tensor
        The input data

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.nn.silu(x._expr), name)


def gelu(x: Tensor, approximate: Optional[str] = None, name: str = "gelu") -> Tensor:
    r"""Applies the Gaussian Error Linear Units function

    .. math::
        \text{GeLU}(x) = 0.5 * x * (1 + \text{erf}(x * 0.5**0.5))

    where :math:`erf` is the Gauss Error function.

    Parameters
    ----------
    x : Tensor
        The input data

    approximate : Optional[str]
        If set to tanh, use an approximation when calculating CDF.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    dtype = x._expr.struct_info.dtype
    if approximate == "tanh":
        tanh_const = rx.const(1 + np.tanh(np.sqrt(2 / np.pi)), dtype=dtype)
        gelu_out = (
            rx.const(0.5, dtype)
            * x._expr
            * (
                tanh_const
                * (x._expr + (rx.const(0.044715, dtype) * _op.power(x._expr, rx.const(3, "int32"))))
            )
        )
    else:
        gelu_out = _op.nn.gelu(x._expr)
    return _wrap_nested(gelu_out, name)


def softmax(x: Tensor, axis: int = -1, name: str = "softmax") -> Tensor:
    r"""Computes softmax.

    .. math:: \text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Parameters
    ----------
    data: Tensor
        The input data to the operator.

    axis: int
        The axis to sum over when computing softmax.
        If not specified, it is by default the last axis of the input tensor.
        Supports negative indexing.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.nn.softmax(x._expr, axis), name)


def dropout(x: Tensor, p: float = 0.5, name: str = "dropout") -> Tensor:
    """Applies the dropout operation to the input tensor.

    During training, each element of the input is set to zero with
    probability ``p``. The whole array is scaled by ``1/(1-p)``
    to keep the expected sum of the input unchanged.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    p : float
        The probability for an element to be reset to 0.

    Returns
    -------
    result : relax.Expr
        The result of dropout, which is a tuple of two tensors.
        The first one is the original tensor and the second one is a
        mask tensor (1.0 where element not dropped, 0.0 where dropped)
    """
    # droped_tup = _op.nn.dropout(x._expr, p)
    # return _wrap_nested(rx.TupleGetItem(droped_tup, 1), name)

    return x


def sigmoid(x, name="sigmoid") -> Tensor:
    """Compute element-wise sigmoid of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.sigmoid(x._expr), name)


def layer_norm(
    x: Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    name: str = "layer_norm",
) -> Tensor:
    r"""
    Layer normalization (Lei Ba and et al., 2016).
    Applies layer normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}}
            * gamma + beta

    Unlike batch normalization, the mean and var are computed along the channel dimension.

    Assume the input has size k on axis 1, then both gamma and beta have shape (k,).

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    x : Tensor
        Input to which layer_norm will be applied.

    normalized_shape: Union[int, List[int]]
        The shape of axes to normalize. If a single integer
        is used, it is treated as a singleton list and this
        module will normalize over the last dimension.

    weight: Tensor
        The gamma scale factor.

    bias: Tensor
        The beta offset factor.

    eps: float
        Small float added to variance to avoid dividing by zero.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    dim_num = len(normalized_shape)
    axes = list(range(-dim_num, 0))
    dtype = x._expr.struct_info.dtype

    if weight is not None:
        weight = weight._expr
    else:
        weight = rx.const(np.ones(normalized_shape), dtype=dtype)
    if bias is not None:
        bias = bias._expr
    else:
        bias = rx.const(np.zeros(normalized_shape), dtype=dtype)

    return _wrap_nested(
        _op.nn.layer_norm(
            x._expr,
            gamma=weight,
            beta=bias,
            axes=axes,
            epsilon=eps,
        ),
        name=name,
    )


def rms_norm(
    x: Tensor,
    weight: Tensor,
    axes: Union[int, List[int]],
    epsilon: float = 1e-5,
    name: str = "rms_norm",
) -> Tensor:
    r"""
    Root mean square normalization (Biao Zhang and et al., 2019).
    Applies root mean square normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

        out = \frac{data}{\sqrt{mean(data, axis)+\epsilon}} * weight

    Parameters
    ----------
    data : Tensor
        Input to which rms_norm will be applied.

    weight : Tensor
        The scale factor.

    axes : Union[int, List[int]]
        The axes that along which the normalization is applied.

    epsilon : float
        Small float added to square mean to avoid dividing by zero.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return _wrap_nested(_op.nn.rms_norm(x._expr, weight._expr, axes, epsilon), name)


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float = 1e-5,
    channel_axis: int = 1,
    axes: Optional[List[int]] = None,
    name: str = "group_norm",
) -> Tensor:
    r"""
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Parameters
    ----------
    x : Tensor
        Input to which rms_norm will be applied.

    num_groups : int
        Number of groups to separate the channels into.

    weight : Tensor
        The gamma scale factor.

    bias : Tensor
        The beta offset factor.

    epsilon : float
        Small float added to square mean to avoid dividing by zero.

    channel_axis: int
        The channel axis of the data.

    axes : Optional[int]
        Which axes to compute the groupnorm over. If None, assumes first
        two channels should be ignored.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    if weight is not None:
        weight = weight._expr
    if bias is not None:
        bias = bias._expr
    dim = len(x._expr.struct_info.shape)
    if axes is None:
        axes = list(range(2, dim))
    return _wrap_nested(
        _op.nn.group_norm(
            x._expr,
            weight,
            bias,
            num_groups,
            channel_axis=channel_axis,
            axes=axes,
            epsilon=eps,
        ),
        name,
    )


def triu(x: Tensor, diagonal: int = 0, name: str = "triu") -> Tensor:
    """Return the upper triangular part of a matrix or a batch of matrices.

    Parameters
    ----------
    x : Tensor
        The tensor that triu will be applied to.
        It is required to have at least two dimensions.

    k : int
        The index indicating the diagonal below which to zero elements.
        If k = 0, the diagonal is the main diagonal.
        If k < 0, the diagonal is below the main diagonal.
        If k > 0, the diagonal is above the main diagonal.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The result tensor.
    """
    return _wrap_nested(_op.triu(x._expr, diagonal), name)


def full(
    shape: Sequence[IntExpr],
    fill_value: Tensor,
    dtype: str = "float32",
    name: str = "full",
) -> Tensor:
    """Fill array with scalar value.

    Parameters
    ----------
    shape : Sequence[IntExpr]
        The shape of the created tensor.

    fill_value : Tensor
        The value to fill. Must be a scalar tensor.

    dtype : str
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of fill_value.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    if isinstance(fill_value, (_tir.FloatImm, _tir.IntImm)):
        fill_value = rx.const(fill_value.value, dtype=dtype)
    elif isinstance(fill_value, (int, float)):
        fill_value = rx.const(fill_value, dtype=dtype)
    else:
        fill_value = fill_value._expr
    return _wrap_nested(_op.full(shape, fill_value, dtype), name)


def zeros(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "zeros",
) -> Tensor:
    """Construct a tensor of all zeros, with the input shape and dtype.

    Parameters
    ----------
    shape : Sequence[IntExpr]
        The shape of the created tensor.

    dtype : str
        The data type of the created tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    zero_expr = _op.zeros(shape, dtype)
    return _wrap_nested(zero_expr, name)
    # return _wrap_nested(_op.triu(x._expr, diagonal), name)
    # return _wrap_nested(_op.zeros(shape, dtype), name)


def ones(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "ones",
) -> Tensor:
    """Construct a tensor of all ones, with the input shape and dtype.

    Parameters
    ----------
    shape : Sequence[IntExpr]
        The shape of the created tensor.

    dtype : str
        The data type of the created tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    return _wrap_nested(_op.ones(shape, dtype), name)


def ones_like(
    x: Tensor,
    dtype: str = "float32",
    name: str = "ones_like",
) -> Tensor:
    """Construct a tensor with all ones, with shape of the input tensor shape.

    Parameters
    ----------
    x : Tensor
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[Union[str, DataType]]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    return _wrap_nested(_op.ones_like(x._expr, dtype), name)


def split(
    ary: Tensor,
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0,
    name: str = "split",
) -> Tuple[Tensor, ...]:
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : Tensor
        Input tensor to be split.
    indices_or_sections : Union[int, Sequence[int]]
        Indices or sections to split into.
    axis : int = 0
        The axis along which to split, default is 0.
    name : str
        Name hint.

    Returns
    -------
    result : Tuple[Tensor, ...]
        A list of sub-arrays as the outcome of splitting.
    """
    return _wrap_nested(_op.split(ary._expr, indices_or_sections, axis), name)


def unbind(
    x: Tensor,
    axis: int = 0,
    name: str = "unbind",
) -> Tensor:
    selections = x.shape[axis]
    res_split = split(x, selections, axis)
    ret = []
    for i in range(selections):
        ret.append(_wrap_nested(_op.squeeze(res_split[i]._expr, axis=[axis]), name))
    return ret


def pad(
    x: Tensor,
    pad: List[int],
    mode: str = "constant",
    value: int = 0,
    name: str = "pad",
) -> Tensor:
    """
    Apply spatial padding to the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor to be padded.
    pad : List[int]
        List in the format of [before_0, after_0, before_1, after_1, ...]
        indicating how much to pad each axis of x.
    mod : str
        Padding mode to use, constant implies padded elements will use
        value argument.
    value : int
        What to pad with in constant mode.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Padded output tensor.
    """
    return _wrap_nested(_op.nn.pad(x._expr, pad_width=pad, pad_value=value, pad_mode=mode), name)


def arange(
    start,
    end=None,
    step=1,
    dtype=None,
    name: str = "arange",
) -> Tensor:
    """Construct a tensor with evenly spaced elements.

    Parameters
    ----------
    start : Union[PrimExprLike,PrimValue]
        The start of the interval.

    end : Optional[Union[PrimExprLike,PrimValue]]
        The end of the interval. If not given, it will be set to start,
        and start will be set to 0.

    step : Union[PrimExprLike,PrimValue]
        The step size.

    dtype : Optional[Union[str, DataType]]
        The data type of the created tensor.

    name : str
        Name hint.


    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    # TODO (yongwww): Update type annotation

    return _wrap_nested(_op.arange(start, end, step, dtype), name)


def get_timestep_embedding(
    x: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    name: str = "get_timestep_embedding",
) -> Tensor:
    """
    Timestep calculation as described in Denoising Diffusion Probabilistic Models.

    Parameters
    ----------
    x : Tensor
        A 1-D Tensor of N indices.
    embedding_dim : int
        The dimension of the output.
    flip_sin_to_cos : bool
        If True, change the order of sine and cosine embeddings.
    downscale_freq_shift : float
        Adjusts the frequency of the sinusoidal sampling.
    scale : float
        Weight adjustment for embedding magnitude.
    max_period : int
        Controls the minimum frequency of the embeddings.
    name : str
        The name to label this operator with.

    Returns
    -------
    result : Tensor
        [N x dim] Tensor of positional embeddings.
    """
    dtype = get_default_dtype()

    # Arithmetic should be done in float for precision.
    timesteps = _op.astype(x._expr, "float32")

    half_dim = embedding_dim // 2
    exponent = rx.const(-math.log(max_period), "float32") * _op.arange(
        start=0, end=half_dim, dtype="float32"
    )
    exponent = exponent / (rx.const(half_dim - downscale_freq_shift, "float32"))

    emb = _op.exp(exponent)
    emb = _op.expand_dims(timesteps, 1) * _op.expand_dims(emb, 0)
    # Scale embeddings
    if scale != 1:
        emb = rx.const(scale, "float32") * emb

    # Concat sine and cosine embeddings.
    if flip_sin_to_cos:
        emb = _op.concat([_op.cos(emb), _op.sin(emb)], axis=-1)
    else:
        emb = _op.concat([_op.sin(emb), _op.cos(emb)], axis=-1)

    # Zero pad
    if embedding_dim % 2 == 1:
        emb = _op.nn.pad(emb, (0, 1, 0, 0))

    # Cast to proper output type
    emb = _op.astype(emb, dtype)
    return _wrap_nested(emb, name)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = False,
    scale: Optional[float] = None,
    name: str = "scaled_dot_product_attention",
):
    """
    Computes a scaled dot product attention on provided attention
    query, key, and values. Compliant with the functional torch implementation.

    Parameters
    ----------
    query : Tensor
        Tensor representing current attention lookup.
    key : Tensor
        Tensor representing cross attention mapping.
    value : Tensor
        Tensor representing embedded attention values.
    attn_mask : Optional[Tensor]
        Optional mask for attention, not yet supported.
    is_causal : Optional[bool]
        If set, uses a causal attention mask.
    scale : Optional[float]
        Optional extra scaling argument applied to attention.
    name : str
        Name hint for this function.
    """
    assert attn_mask is None, "attn_mask not yet supported."
    causal_mask = "TopLeft" if is_causal else None

    attn = _op.nn.attention(
        query._expr, key._expr, value._expr, causal_mask=causal_mask, scale=scale
    )
    return _wrap_nested(attn, name)


def interpolate(
    x: Tensor,
    size: Optional[Union[int, Tuple[int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: Optional[bool] = None,
    name: str = "interpolate",
):
    """Resize a tensor using the specified mode.

    Parameters
    ----------
    x : Tensor
        Input tensor to be resized.
    size : Optional[Union[int, Tuple[int]]]
        Requested output size, only one of size and scale_factor may
        be specified.
    scale_factor : Optional[Union[float, Tuple[float]]]
        Multiplier for spatial size.
    mode : str
        Algorithm used for sampling.
    align_corners : Optional[bool]
        How to map pixels before and after sampling.
    recompute_scale_factor : Optional[bool]
        Recompute the scale_factor for use in interpolation.
    antialias : Optional[bool]
        Apply antialiasing to output.
    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
        Output tensor with requested shape.
    """
    assert recompute_scale_factor is None, "recompute_scale_factor is not supported."
    assert antialias is None, "antialias is not supported."

    if size is None:
        shape = x.shape
        if isinstance(scale_factor, (list, tuple)):
            size = tuple(int(shape[i] * scale_factor[i]) for i in range(2, len(shape)))
        else:
            size = tuple(int(shape[i] * scale_factor) for i in range(2, len(shape)))

    if mode.startswith("nearest"):
        mode = "nearest_neighbor"
    elif mode[0:2] == "bi":
        mode = mode[2:]

    if mode == "nearest_neighbor":
        coord_trans = "asymmetric"
    elif align_corners:
        coord_trans = "align_corners"
    else:
        coord_trans = "half_pixel"

    return _wrap_nested(
        _op.image.resize2d(
            x._expr,
            size,
            layout="NCHW",
            method=mode,
            coordinate_transformation_mode=coord_trans,
        ),
        name,
    )


def tensor_expr_op(
    tensor_expr_func: Callable,
    name_hint: str,
    args: List[Union[Tensor, _tir.Var, int]],
    *,
    attrs: Optional[Dict[str, Any]] = None,
):
    """Build the given tensor_expr_func with te.

    Parameters
    ----------
    tensor_expr_func : Callable
        A function that returns a te tensor or a list of tensors.

    name_hint : str
        Name hint.

    args: List[Union[Tensor, _tir.Var]]
        Arguments passed to the function.

    attrs: Optional[Dict[str, Any]]
        A dict of attributes to apply to the function.

    Returns
    -------
    result : Tensor
        The result tensor.
    """

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr  # pylint: disable=protected-access
        return arg

    return _wrap_nested(
        BlockBuilder.current().emit_te(
            tensor_expr_func,
            *[_convert(arg) for arg in args],
            primfunc_name_hint=name_hint,
            primfunc_attrs=attrs,
        ),
        name=name_hint,
    )


def print_(array: Tensor):
    if SpecBuilder.current().io_effect is None:
        raise RuntimeError("Printing is only supported when debug mode is on.")
    SpecBuilder.current().io_effect.print_(array)
