"""
This type stub file was generated by pyright.
"""

from collections.abc import Sequence
from typing import Any, Literal, SupportsIndex, TypeVar, overload, TypeAlias
from numpy import _ModeKind, _OrderACF, _OrderKACF, _PartitionKind, _SortKind, _SortSide, bool_, complexfloating, float16, floating, generic, int64, int_, intp, number, object_, uint64
# from numpy._typing import ArrayLike, DTypeLike, NDArray
 # , _ArrayLike, _ArrayLikeBool_co, _ArrayLikeComplex_co, _ArrayLikeFloat_co, _ArrayLikeInt_co, _ArrayLikeObject_co, _ArrayLikeUInt_co, _BoolLike_co, _ComplexLike_co, _DTypeLike, _IntLike_co, _NumberLike_co, _ScalarLike_co, _Shape, _ShapeLike

ArrayLike: TypeAlias = Any
DTypeLike: TypeAlias = Any
NDArray: TypeAlias = Any
_ArrayLike: TypeAlias = Any
_ArrayLikeBool_co: TypeAlias = Any
_ArrayLikeComplex_co: TypeAlias = Any
_ArrayLikeFloat_co: TypeAlias = Any
_ArrayLikeInt_co: TypeAlias = Any
_ArrayLikeObject_co: TypeAlias = Any
_ArrayLikeUInt_co: TypeAlias = Any
_BoolLike_co: TypeAlias = Any
_ComplexLike_co: TypeAlias = Any
_DTypeLike: TypeAlias = Any
_IntLike_co: TypeAlias = Any
_NumberLike_co: TypeAlias = Any
_ScalarLike_co: TypeAlias = Any
_Shape: TypeAlias = Any
_ShapeLike: TypeAlias = Any

_SCT = TypeVar("_SCT", bound=generic)
_SCT_uifcO = TypeVar("_SCT_uifcO", bound=number[Any] | object_)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])
__all__: list[str]
@overload
def take(a: _ArrayLike[_SCT], indices: _IntLike_co, axis: None = ..., out: None = ..., mode: _ModeKind = ...) -> _SCT:
    ...

@overload
def take(a: ArrayLike, indices: _IntLike_co, axis: None | SupportsIndex = ..., out: None = ..., mode: _ModeKind = ...) -> Any:
    ...

@overload
def take(a: _ArrayLike[_SCT], indices: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., out: None = ..., mode: _ModeKind = ...) -> NDArray[_SCT]:
    ...

@overload
def take(a: ArrayLike, indices: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., out: None = ..., mode: _ModeKind = ...) -> NDArray[Any]:
    ...

@overload
def take(a: ArrayLike, indices: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., out: _ArrayType = ..., mode: _ModeKind = ...) -> _ArrayType:
    ...

@overload
def reshape(a: _ArrayLike[_SCT], newshape: _ShapeLike, order: _OrderACF = ...) -> NDArray[_SCT]:
    ...

@overload
def reshape(a: ArrayLike, newshape: _ShapeLike, order: _OrderACF = ...) -> NDArray[Any]:
    ...

@overload
def choose(a: _IntLike_co, choices: ArrayLike, out: None = ..., mode: _ModeKind = ...) -> Any:
    ...

@overload
def choose(a: _ArrayLikeInt_co, choices: _ArrayLike[_SCT], out: None = ..., mode: _ModeKind = ...) -> NDArray[_SCT]:
    ...

@overload
def choose(a: _ArrayLikeInt_co, choices: ArrayLike, out: None = ..., mode: _ModeKind = ...) -> NDArray[Any]:
    ...

@overload
def choose(a: _ArrayLikeInt_co, choices: ArrayLike, out: _ArrayType = ..., mode: _ModeKind = ...) -> _ArrayType:
    ...

@overload
def repeat(a: _ArrayLike[_SCT], repeats: _ArrayLikeInt_co, axis: None | SupportsIndex = ...) -> NDArray[_SCT]:
    ...

@overload
def repeat(a: ArrayLike, repeats: _ArrayLikeInt_co, axis: None | SupportsIndex = ...) -> NDArray[Any]:
    ...

def put(a: NDArray[Any], ind: _ArrayLikeInt_co, v: ArrayLike, mode: _ModeKind = ...) -> None:
    ...

@overload
def swapaxes(a: _ArrayLike[_SCT], axis1: SupportsIndex, axis2: SupportsIndex) -> NDArray[_SCT]:
    ...

@overload
def swapaxes(a: ArrayLike, axis1: SupportsIndex, axis2: SupportsIndex) -> NDArray[Any]:
    ...

@overload
def transpose(a: _ArrayLike[_SCT], axes: None | _ShapeLike = ...) -> NDArray[_SCT]:
    ...

@overload
def transpose(a: ArrayLike, axes: None | _ShapeLike = ...) -> NDArray[Any]:
    ...

@overload
def partition(a: _ArrayLike[_SCT], kth: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., kind: _PartitionKind = ..., order: None | str | Sequence[str] = ...) -> NDArray[_SCT]:
    ...

@overload
def partition(a: ArrayLike, kth: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., kind: _PartitionKind = ..., order: None | str | Sequence[str] = ...) -> NDArray[Any]:
    ...

def argpartition(a: ArrayLike, kth: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., kind: _PartitionKind = ..., order: None | str | Sequence[str] = ...) -> NDArray[intp]:
    ...

@overload
def sort(a: _ArrayLike[_SCT], axis: None | SupportsIndex = ..., kind: None | _SortKind = ..., order: None | str | Sequence[str] = ...) -> NDArray[_SCT]:
    ...

@overload
def sort(a: ArrayLike, axis: None | SupportsIndex = ..., kind: None | _SortKind = ..., order: None | str | Sequence[str] = ...) -> NDArray[Any]:
    ...

def argsort(a: Any, axis: Any = ..., kind: Any = ..., order: Any = ...) -> NDArray[intp]:
    ...

@overload
def argmax(a: ArrayLike, axis: None = ..., out: None = ..., *, keepdims: Literal[False] = ...) -> intp:
    ...

@overload
def argmax(a: ArrayLike, axis: None | SupportsIndex = ..., out: None = ..., *, keepdims: bool = ...) -> Any:
    ...

@overload
def argmax(a: ArrayLike, axis: None | SupportsIndex = ..., out: _ArrayType = ..., *, keepdims: bool = ...) -> _ArrayType:
    ...

@overload
def argmin(a: ArrayLike, axis: None = ..., out: None = ..., *, keepdims: Literal[False] = ...) -> intp:
    ...

@overload
def argmin(a: ArrayLike, axis: None | SupportsIndex = ..., out: None = ..., *, keepdims: bool = ...) -> Any:
    ...

@overload
def argmin(a: ArrayLike, axis: None | SupportsIndex = ..., out: _ArrayType = ..., *, keepdims: bool = ...) -> _ArrayType:
    ...

@overload
def searchsorted(a: Any, v: Any, side: Any = ..., sorter: None | _ArrayLikeInt_co = ...) -> intp:
    ...

@overload
def searchsorted(a: Any, v: Any, side: Any = ..., sorter: None | Any = ...) -> NDArray[intp]:
    ...

@overload
def resize(a: _ArrayLike[_SCT], new_shape: _ShapeLike) -> NDArray[_SCT]:
    ...

@overload
def resize(a: ArrayLike, new_shape: _ShapeLike) -> NDArray[Any]:
    ...

@overload
def squeeze(a: _SCT, axis: None | _ShapeLike = ...) -> _SCT:
    ...

@overload
def squeeze(a: _ArrayLike[_SCT], axis: None | _ShapeLike = ...) -> NDArray[_SCT]:
    ...

@overload
def squeeze(a: ArrayLike, axis: None | _ShapeLike = ...) -> NDArray[Any]:
    ...

@overload
def diagonal(a: _ArrayLike[_SCT], offset: SupportsIndex = ..., axis1: SupportsIndex = ..., axis2: SupportsIndex = ...) -> NDArray[_SCT]:
    ...

@overload
def diagonal(a: ArrayLike, offset: SupportsIndex = ..., axis1: SupportsIndex = ..., axis2: SupportsIndex = ...) -> NDArray[Any]:
    ...

@overload
def trace(a: ArrayLike, offset: SupportsIndex = ..., axis1: SupportsIndex = ..., axis2: SupportsIndex = ..., dtype: DTypeLike = ..., out: None = ...) -> Any:
    ...

@overload
def trace(a: ArrayLike, offset: SupportsIndex = ..., axis1: SupportsIndex = ..., axis2: SupportsIndex = ..., dtype: DTypeLike = ..., out: _ArrayType = ...) -> _ArrayType:
    ...

@overload
def ravel(a: _ArrayLike[_SCT], order: _OrderKACF = ...) -> NDArray[_SCT]:
    ...

@overload
def ravel(a: ArrayLike, order: _OrderKACF = ...) -> NDArray[Any]:
    ...

def nonzero(a: ArrayLike) -> tuple[NDArray[intp], ...]:
    ...

def shape(a: ArrayLike) -> _Shape:
    ...

@overload
def compress(condition: _ArrayLikeBool_co, a: _ArrayLike[_SCT], axis: None | SupportsIndex = ..., out: None = ...) -> NDArray[_SCT]:
    ...

@overload
def compress(condition: _ArrayLikeBool_co, a: ArrayLike, axis: None | SupportsIndex = ..., out: None = ...) -> NDArray[Any]:
    ...

@overload
def compress(condition: _ArrayLikeBool_co, a: ArrayLike, axis: None | SupportsIndex = ..., out: _ArrayType = ...) -> _ArrayType:
    ...

@overload
def clip(a: _SCT, a_min: None | ArrayLike, a_max: None | ArrayLike, out: None = ..., *, dtype: None = ..., where: None | _ArrayLikeBool_co = ..., order: _OrderKACF = ..., subok: bool = ..., signature: str | tuple[None | str, ...] = ..., extobj: list[Any] = ...) -> _SCT:
    ...

@overload
def clip(a: _ScalarLike_co, a_min: None | ArrayLike, a_max: None | ArrayLike, out: None = ..., *, dtype: None = ..., where: None | _ArrayLikeBool_co = ..., order: _OrderKACF = ..., subok: bool = ..., signature: str | tuple[None | str, ...] = ..., extobj: list[Any] = ...) -> Any:
    ...

@overload
def clip(a: _ArrayLike[_SCT], a_min: None | ArrayLike, a_max: None | ArrayLike, out: None = ..., *, dtype: None = ..., where: None | _ArrayLikeBool_co = ..., order: _OrderKACF = ..., subok: bool = ..., signature: str | tuple[None | str, ...] = ..., extobj: list[Any] = ...) -> NDArray[_SCT]:
    ...

@overload
def clip(a: ArrayLike, a_min: None | ArrayLike, a_max: None | ArrayLike, out: None = ..., *, dtype: None = ..., where: None | _ArrayLikeBool_co = ..., order: _OrderKACF = ..., subok: bool = ..., signature: str | tuple[None | str, ...] = ..., extobj: list[Any] = ...) -> NDArray[Any]:
    ...

@overload
def clip(a: ArrayLike, a_min: None | ArrayLike, a_max: None | ArrayLike, out: _ArrayType = ..., *, dtype: DTypeLike, where: None | _ArrayLikeBool_co = ..., order: _OrderKACF = ..., subok: bool = ..., signature: str | tuple[None | str, ...] = ..., extobj: list[Any] = ...) -> Any:
    ...

@overload
def clip(a: ArrayLike, a_min: None | ArrayLike, a_max: None | ArrayLike, out: _ArrayType, *, dtype: DTypeLike = ..., where: None | _ArrayLikeBool_co = ..., order: _OrderKACF = ..., subok: bool = ..., signature: str | tuple[None | str, ...] = ..., extobj: list[Any] = ...) -> _ArrayType:
    ...

@overload
def sum(a: _ArrayLike[_SCT], axis: None = ..., dtype: None = ..., out: None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def sum(a: ArrayLike, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def sum(a: ArrayLike, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def all(a: ArrayLike, axis: None = ..., out: None = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> bool_:
    ...

@overload
def all(a: ArrayLike, axis: None | _ShapeLike = ..., out: None = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def all(a: ArrayLike, axis: None | _ShapeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def any(a: ArrayLike, axis: None = ..., out: None = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> bool_:
    ...

@overload
def any(a: ArrayLike, axis: None | _ShapeLike = ..., out: None = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def any(a: ArrayLike, axis: None | _ShapeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def cumsum(a: Any, axis: None | Any = ..., dtype: None = ..., out: None = ...) -> NDArray[_SCT]:
    ...

@overload
def cumsum(a: Any, axis: None | Any = ..., dtype: None = ..., out: None = ...) -> NDArray[Any]:
    ...

@overload
def cumsum(a: Any, axis: None | Any = ..., dtype: Any = ..., out: None = ...) -> NDArray[_SCT]:
    ...

@overload
def cumsum(a: Any, axis: None | Any = ..., dtype: Any = ..., out: None = ...) -> NDArray[Any]:
    ...

@overload
def cumsum(a: Any, axis: None | Any = ..., dtype: Any = ..., out: Any = ...) -> _ArrayType:
    ...

@overload
def ptp(a: _ArrayLike[_SCT], axis: None = ..., out: None = ..., keepdims: Literal[False] = ...) -> _SCT:
    ...

@overload
def ptp(a: ArrayLike, axis: None | _ShapeLike = ..., out: None = ..., keepdims: bool = ...) -> Any:
    ...

@overload
def ptp(a: ArrayLike, axis: None | _ShapeLike = ..., out: _ArrayType = ..., keepdims: bool = ...) -> _ArrayType:
    ...

@overload
def amax(a: _ArrayLike[_SCT], axis: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def amax(a: ArrayLike, axis: None | _ShapeLike = ..., out: None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def amax(a: ArrayLike, axis: None | _ShapeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def amin(a: _ArrayLike[_SCT], axis: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def amin(a: ArrayLike, axis: None | _ShapeLike = ..., out: None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def amin(a: ArrayLike, axis: None | _ShapeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def prod(a: _ArrayLikeBool_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> int_:
    ...

@overload
def prod(a: _ArrayLikeUInt_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> uint64:
    ...

@overload
def prod(a: _ArrayLikeInt_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> int64:
    ...

@overload
def prod(a: _ArrayLikeFloat_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> floating[Any]:
    ...

@overload
def prod(a: _ArrayLikeComplex_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> complexfloating[Any, Any]:
    ...

@overload
def prod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: None = ..., out: None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def prod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None = ..., dtype: _DTypeLike[_SCT] = ..., out: None = ..., keepdims: Literal[False] = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def prod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: None | DTypeLike = ..., out: None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def prod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: None | DTypeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def cumprod(a: _ArrayLikeBool_co, axis: None | SupportsIndex = ..., dtype: None = ..., out: None = ...) -> NDArray[int_]:
    ...

@overload
def cumprod(a: _ArrayLikeUInt_co, axis: None | SupportsIndex = ..., dtype: None = ..., out: None = ...) -> NDArray[uint64]:
    ...

@overload
def cumprod(a: _ArrayLikeInt_co, axis: None | SupportsIndex = ..., dtype: None = ..., out: None = ...) -> NDArray[int64]:
    ...

@overload
def cumprod(a: _ArrayLikeFloat_co, axis: None | SupportsIndex = ..., dtype: None = ..., out: None = ...) -> NDArray[floating[Any]]:
    ...

@overload
def cumprod(a: _ArrayLikeComplex_co, axis: None | SupportsIndex = ..., dtype: None = ..., out: None = ...) -> NDArray[complexfloating[Any, Any]]:
    ...

@overload
def cumprod(a: _ArrayLikeObject_co, axis: None | SupportsIndex = ..., dtype: None = ..., out: None = ...) -> NDArray[object_]:
    ...

@overload
def cumprod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | SupportsIndex = ..., dtype: _DTypeLike[_SCT] = ..., out: None = ...) -> NDArray[_SCT]:
    ...

@overload
def cumprod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | SupportsIndex = ..., dtype: DTypeLike = ..., out: None = ...) -> NDArray[Any]:
    ...

@overload
def cumprod(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | SupportsIndex = ..., dtype: DTypeLike = ..., out: _ArrayType = ...) -> _ArrayType:
    ...

def ndim(a: ArrayLike) -> int:
    ...

def size(a: ArrayLike, axis: None | int = ...) -> int:
    ...

@overload
def around(a: _BoolLike_co, decimals: SupportsIndex = ..., out: None = ...) -> float16:
    ...

@overload
def around(a: _SCT_uifcO, decimals: SupportsIndex = ..., out: None = ...) -> _SCT_uifcO:
    ...

@overload
def around(a: _ComplexLike_co | object_, decimals: SupportsIndex = ..., out: None = ...) -> Any:
    ...

@overload
def around(a: _ArrayLikeBool_co, decimals: SupportsIndex = ..., out: None = ...) -> NDArray[float16]:
    ...

@overload
def around(a: _ArrayLike[_SCT_uifcO], decimals: SupportsIndex = ..., out: None = ...) -> NDArray[_SCT_uifcO]:
    ...

@overload
def around(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, decimals: SupportsIndex = ..., out: None = ...) -> NDArray[Any]:
    ...

@overload
def around(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, decimals: SupportsIndex = ..., out: _ArrayType = ...) -> _ArrayType:
    ...

@overload
def mean(a: _ArrayLikeFloat_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> floating[Any]:
    ...

@overload
def mean(a: _ArrayLikeComplex_co, axis: None = ..., dtype: None = ..., out: None = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> complexfloating[Any, Any]:
    ...

@overload
def mean(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: None = ..., out: None = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def mean(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None = ..., dtype: _DTypeLike[_SCT] = ..., out: None = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def mean(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: None = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def mean(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _ArrayType = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def std(a: _ArrayLikeComplex_co, axis: None = ..., dtype: None = ..., out: None = ..., ddof: float = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> floating[Any]:
    ...

@overload
def std(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: None = ..., out: None = ..., ddof: float = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def std(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None = ..., dtype: _DTypeLike[_SCT] = ..., out: None = ..., ddof: float = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def std(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def std(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _ArrayType = ..., ddof: float = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

@overload
def var(a: _ArrayLikeComplex_co, axis: None = ..., dtype: None = ..., out: None = ..., ddof: float = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> floating[Any]:
    ...

@overload
def var(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: None = ..., out: None = ..., ddof: float = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def var(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None = ..., dtype: _DTypeLike[_SCT] = ..., out: None = ..., ddof: float = ..., keepdims: Literal[False] = ..., *, where: _ArrayLikeBool_co = ...) -> _SCT:
    ...

@overload
def var(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> Any:
    ...

@overload
def var(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _ArrayType = ..., ddof: float = ..., keepdims: bool = ..., *, where: _ArrayLikeBool_co = ...) -> _ArrayType:
    ...

