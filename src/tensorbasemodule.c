#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>

#include "tensorbase.c"

// https://getcode.substack.com/p/fun-and-hackable-tensors-in-rust
// https://jessicastringham.net/2017/12/31/stride-tricks/
// https://github.com/abeschneider/tensor/blob/master/include/tensor_ops.hpp
// https://github.com/kurtschelfthout/tensorken
// https://www.google.com/search?q=as%20strided%20broadcasting&ie=utf-8&oe=utf-8&client=firefox-b-1-m
// https://www.google.com/search?client=firefox-b-1-m&sca_esv=da8adb7374804231&q=as+strided+convolution&oq=as+strided+convolution&aqs=heirloom-srp..

// TODO:
// - what to do about data types? (void * everywhere?)
// - allow use as base class?
// - iterable
// - implement buffer protocol?
//      - http://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/
//      - https://docs.python.org/3/c-api/buffer.html
// - implement number protocol
// - implement sequence protocol
// - https://docs.python.org/3/c-api/structures.html#c.METH_FASTCALL
//  Py_TPFLAGS_CHECKTYPES
// - indexing and slicing
// - multiple tensors can share the same data
// - squeeze, reshape, resize, transpose, etc.
// - len, getitem, setitem
// - handle case of single item tensors (ndim = 0)
// - be consistent about taking ptr or value

// TODO: useful macros
// - Py_ALWAYS_INLINE
// - Py_MAX(x, y)
// - Py_MIN(x, y)
// - Py_STRINGIFY(x)
// - PyDoc_STRVAR(name, str)
// - PyDoc_STR(str)

// ----------------------------------------------------------------
// ▗▄▄▖           ▗▄▄▄▖                         ▗▄▄▖
// ▐▛▀▜▖          ▝▀█▀▘                         ▐▛▀▜▌
// ▐▌ ▐▌▝█ █▌       █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌▐▌ ▐▌ ▟██▖▗▟██▖ ▟█▙
// ▐██▛  █▖█        █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘  ▐███  ▘▄▟▌▐▙▄▖▘▐▙▄▟▌
// ▐▌    ▐█▛        █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █   ▐▌ ▐▌▗█▀▜▌ ▀▀█▖▐▛▀▀▘
// ▐▌     █▌        █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █   ▐▙▄▟▌▐▙▄█▌▐▄▄▟▌▝█▄▄▌
// ▝▘     █         ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀   ▝▀▀▀  ▀▀▝▘ ▀▀▀  ▝▀▀
//       █▌
// ----------------------------------------------------------------

//
// A wrapper around TensorBase enabling use as a Python object.
// We try to match the pytorch tensor API as closely as possible.
//

// clang-format off
typedef struct
{
    PyObject_HEAD
    TensorBase tb;
} PyTensorBase;
// clang-format on

// ----------------------------------------------------------------
// ▗▄▄▖                                 █
// ▐▛▀▜▖                          ▐▌    ▀
// ▐▌ ▐▌ █▟█▌ ▟█▙ ▐▙█▙  ▟█▙  █▟█▌▐███  ██   ▟█▙ ▗▟██▖
// ▐██▛  █▘  ▐▛ ▜▌▐▛ ▜▌▐▙▄▟▌ █▘   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌    █   ▐▌ ▐▌▐▌ ▐▌▐▛▀▀▘ █    ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▐▌    █   ▝█▄█▘▐█▄█▘▝█▄▄▌ █    ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
// ▝▘    ▀    ▝▀▘ ▐▌▀▘  ▝▀▀  ▀     ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
//                ▐▌
// ----------------------------------------------------------------

static PyObject *PyTensorBase_get_ndim(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->tb.ndim);
}

static PyObject *PyTensorBase_get_shape(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *shape = PyTuple_New(self->tb.ndim);

    for (long i = 0; i < self->tb.ndim; i++)
    {
        if (PyTuple_SetItem(shape, i, PyLong_FromLong(self->tb.shape[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return shape;
}

static PyGetSetDef PyTensorBase_getset[] = {
    {"ndim", (getter)PyTensorBase_get_ndim, NULL, "TODO: docs", NULL},
    {"shape", (getter)PyTensorBase_get_shape, NULL, "TODO: docs", NULL},
    // {"__getitem__", (getter)PyTensorBase_get_item, NULL, "TODO: docs", NULL},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
// ▗▄ ▄▖          ▗▖           ▗▖
// ▐█ █▌      ▐▌  ▐▌           ▐▌
// ▐███▌ ▟█▙ ▐███ ▐▙██▖ ▟█▙  ▟█▟▌▗▟██▖
// ▐▌█▐▌▐▙▄▟▌ ▐▌  ▐▛ ▐▌▐▛ ▜▌▐▛ ▜▌▐▙▄▖▘
// ▐▌▀▐▌▐▛▀▀▘ ▐▌  ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌ ▀▀█▖
// ▐▌ ▐▌▝█▄▄▌ ▐▙▄ ▐▌ ▐▌▝█▄█▘▝█▄█▌▐▄▄▟▌
// ▝▘ ▝▘ ▝▀▀   ▀▀ ▝▘ ▝▘ ▝▀▘  ▝▀▝▘ ▀▀▀
// ----------------------------------------------------------------

static PyObject *PyTensorBase_numel(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->tb.numel);
}

static PyObject *PyTensorBase_stride(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *stride = PyTuple_New(self->tb.ndim);

    for (long i = 0; i < self->tb.ndim; i++)
    {
        if (PyTuple_SetItem(stride, i, PyLong_FromLong(self->tb.strides[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return stride;
}

static PyMethodDef PyTensorBase_methods[] = {
    {"numel", (PyCFunction)PyTensorBase_numel, METH_NOARGS, "TODO: docs"},
    {"stride", (PyCFunction)PyTensorBase_stride, METH_NOARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
// ▗▄ ▗▖          ▗▖                  ▗▄▄▖                               ▗▄▖
// ▐█ ▐▌          ▐▌                  ▐▛▀▜▖           ▐▌                 ▝▜▌
// ▐▛▌▐▌▐▌ ▐▌▐█▙█▖▐▙█▙  ▟█▙  █▟█▌     ▐▌ ▐▌ █▟█▌ ▟█▙ ▐███  ▟█▙  ▟██▖ ▟█▙  ▐▌
// ▐▌█▐▌▐▌ ▐▌▐▌█▐▌▐▛ ▜▌▐▙▄▟▌ █▘       ▐██▛  █▘  ▐▛ ▜▌ ▐▌  ▐▛ ▜▌▐▛  ▘▐▛ ▜▌ ▐▌
// ▐▌▐▟▌▐▌ ▐▌▐▌█▐▌▐▌ ▐▌▐▛▀▀▘ █        ▐▌    █   ▐▌ ▐▌ ▐▌  ▐▌ ▐▌▐▌   ▐▌ ▐▌ ▐▌
// ▐▌ █▌▐▙▄█▌▐▌█▐▌▐█▄█▘▝█▄▄▌ █        ▐▌    █   ▝█▄█▘ ▐▙▄ ▝█▄█▘▝█▄▄▌▝█▄█▘ ▐▙▄
// ▝▘ ▀▘ ▀▀▝▘▝▘▀▝▘▝▘▀▘  ▝▀▀  ▀        ▝▘    ▀    ▝▀▘   ▀▀  ▝▀▘  ▝▀▀  ▝▀▘   ▀▀
// ----------------------------------------------------------------

// These methods are implemented below
// TODO: move implementations here?
static PyObject *PyTensorBase_add(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_divide(PyObject *a, PyObject *b);
static PyObject *PyTensorBase_negate(PyObject *a);

static PyObject *PyTensorBase_matrix_multiply(PyTensorBase *a, PyTensorBase *b);

static PyNumberMethods PyTensorBase_as_number = {
    .nb_add = (binaryfunc)PyTensorBase_add,
    // .nb_subtract = (binaryfunc)PyTensorBase_as_number_subtract,
    // .nb_multiply = (binaryfunc)PyTensorBase_as_number_multiply,
    // .nb_floor_divide = (binaryfunc)PyTensorBase_as_number_floor_divide,
    .nb_true_divide = (binaryfunc)PyTensorBase_divide,
    // .nb_remainder = (binaryfunc)PyTensorBase_as_number_remainder,
    // .nb_divmod = (binaryfunc)PyTensorBase_

    .nb_matrix_multiply = (binaryfunc)PyTensorBase_matrix_multiply,

    // .nb_power = (ternaryfunc)PyTensorBase_as_number_power,

    .nb_negative = (unaryfunc)PyTensorBase_negate,
    // .nb_positive = (unaryfunc)PyTensorBase_as_number_positive,
    // .nb_absolute = (unaryfunc)PyTensorBase_as_number_absolute,
    // .nb_invert = (unaryfunc)PyTensorBase_as_number_invert,

    // .nb_lshift = (binaryfunc)PyTensorBase_as_number_lshift,
    // .nb_rshift = (binaryfunc)PyTensorBase_as_number_rshift,

    // .nb_bool = (inquiry)PyTensorBase_,

    // .nb_and = (binaryfunc)PyTensorBase_as_number_and,
    // .nb_xor = (binaryfunc)PyTensorBase_as_number_xor,
    // .nb_or = (binaryfunc)PyTensorBase_as_number_or,

    // .nb_int = (unaryfunc)PyTensorBase_as_number_int,
    // .nb_float = (unaryfunc)PyTensorBase_as_number_float,

    // .nb_inplace_add = (binaryfunc)PyTensorBase_as_number_inplace_add,
    // .nb_inplace_subtract = (binaryfunc)PyTensorBase_as_number_inplace_subtract,
    // .nb_inplace_multiply = (binaryfunc)PyTensorBase_as_number_inplace_multiply,
    // .nb_inplace_floor_divide = (binaryfunc)PyTensorBase_as_number_inplace_floor_divide,
    // .nb_inplace_true_divide = (binaryfunc)PyTensorBase_as_number_inplace_true_divide,
    // .nb_inplace_remainder = (binaryfunc)PyTensorBase_as_number_inplace_remainder,
    // .nb_inplace_matrix_multiply = (binaryfunc)PyTensorBase_as_number_inplace_matrix_multiply,
    // .nb_inplace_power = (ternaryfunc)PyTensorBase_as_number_inplace_power,
    // .nb_inplace_lshift = (binaryfunc)PyTensorBase_as_number_inplace_lshift,
    // .nb_inplace_rshift = (binaryfunc
    // .nb_inplace_and
    // .nb_inplace_xor
    // .nb_inplace_or

    // .nb_index
};

// ----------------------------------------------------------------
// ▗▄▄▖            ▗▄▖ ▗▖     █                      ▗▄▄         ▄▄
// ▐▛▀▜▖           █▀█ ▐▌     ▀             ▐▌       ▐▛▀█       ▐▛▀
// ▐▌ ▐▌▝█ █▌     ▐▌ ▐▌▐▙█▙  ██   ▟█▙  ▟██▖▐███      ▐▌ ▐▌ ▟█▙ ▐███
// ▐██▛  █▖█      ▐▌ ▐▌▐▛ ▜▌  █  ▐▙▄▟▌▐▛  ▘ ▐▌       ▐▌ ▐▌▐▙▄▟▌ ▐▌
// ▐▌    ▐█▛      ▐▌ ▐▌▐▌ ▐▌  █  ▐▛▀▀▘▐▌    ▐▌       ▐▌ ▐▌▐▛▀▀▘ ▐▌
// ▐▌     █▌       █▄█ ▐█▄█▘  █  ▝█▄▄▌▝█▄▄▌ ▐▙▄      ▐▙▄█ ▝█▄▄▌ ▐▌
// ▝▘     █        ▝▀▘ ▝▘▀▘   █   ▝▀▀  ▝▀▀   ▀▀      ▝▀▀   ▝▀▀  ▝▘
//       █▌                 ▐█▛
// ----------------------------------------------------------------

static int args_to_shape(PyObject *args, ShapeArray *tb_shape)
{
    // Parse args as tuple of dimensions (or tuple of tuple of dimensions)
    Py_ssize_t tuple_len = PyTuple_Size(args);

    if (tuple_len == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor must have at least one value.");
        return -1;
    }

    if (tuple_len > MAX_RANK)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor rank exceeds maximum allowed.");
        return -1;
    }

    memset(tb_shape, 0, sizeof(ShapeArray));

    for (long i = 0; i < tuple_len; i++)
    {
        PyObject *item = PyTuple_GetItem(args, i);
        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Tensor dimensions must be integers.");
            return -1;
        }

        (*tb_shape)[i] = PyLong_AsLong(item);
    }

    return 1;
}

static int PyTensorBase_init(PyTensorBase *self, PyObject *args, PyObject *kwds)
{

    ShapeArray tb_shape = {0};
    if (args_to_shape(args, &tb_shape) < 0)
    {
        // NOTE: error message set in args_to_shape
        return -1;
    }

    if (TensorBase_init(&self->tb, tb_shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor data.");
        return -1;
    }

    return 0;
}

static void PyTensorBase_dealloc(PyTensorBase *self)
{
    TensorBase_dealloc(&self->tb);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyTensorBase_str(PyTensorBase *obj)
{
    // TODO: calculate a reasonable buffer size
    char *str_buffer = malloc(100 * sizeof(char));

    TensorBase_to_string(&obj->tb, str_buffer, 100 * sizeof(char));

    return Py_BuildValue("s", str_buffer);
}

// clang-format off
// NOTE: disabling formatter due to PyVarObject_HEAD_INIT macro
static PyTypeObject PyTensorBaseType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "match.tensorbase.TensorBase",
    .tp_doc = PyDoc_STR("TODO: docs"),
    // .tp_repr = PyTensorBase_str,
    .tp_str = PyTensorBase_str,
    .tp_basicsize = sizeof(PyTensorBase),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyTensorBase_init,
    .tp_dealloc = (destructor)PyTensorBase_dealloc,
    // .tp_members = PyTensorBase_members,
    .tp_methods = PyTensorBase_methods,
    .tp_getset = PyTensorBase_getset,
    // TODO: .tp_getattr/tp_setattr?
    // TODO: .tp_getattro/tp_setattro?
    // TODO: .tp_richcompare?
    // TODO: .tp_iter/tp_iternext?
    .tp_as_number = &PyTensorBase_as_number,
};
// clang-format on

// ----------------------------------------------------------------
// ▗▖ ▗▖       █  ▗▄▖    █         █
// ▐▌ ▐▌ ▐▌    ▀  ▝▜▌    ▀   ▐▌    ▀
// ▐▌ ▐▌▐███  ██   ▐▌   ██  ▐███  ██   ▟█▙ ▗▟██▖
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▝█▄█▘ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
//  ▝▀▘   ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
// ----------------------------------------------------------------

static long PyTensorBase_Check(PyObject *obj)
{
    return PyObject_IsInstance(obj, (PyObject *)&PyTensorBaseType);
}

static long PyFloatOrLong_Check(PyObject *obj)
{
    return PyLong_Check(obj) || PyFloat_Check(obj);
}

static long can_math(PyObject *obj)
{
    return PyFloatOrLong_Check(obj) || PyTensorBase_Check(obj);
}

static scalar PyFloatOrLong_asDouble(PyObject *obj)
{
    if (PyLong_Check(obj))
    {
        return PyLong_AsDouble(obj);
    }
    return PyFloat_AsDouble(obj);
}

static PyTensorBase *PyTensorBase_create(ShapeArray shape)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    if (TensorBase_init(&result->tb, shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor base.");
        return NULL;
    }

    return result;
}

// static PyTensorBase *PyTensorBase_shallow_broadcast(PyTensorBase *t, ShapeArray shape)
// {
//     PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
//     if (result == NULL)
//     {
//         PyErr_SetString(PyExc_RuntimeError, "Failed to create new PyTensorBase object.");
//         return NULL;
//     }

//     result->tb = t->tb;

//     // Muck with strides...

//     return result;
// }

// ----------------------------------------------------------------
// ▗▄▄▄▖                              ▗▄ ▄▖          ▗▖
// ▝▀█▀▘                              ▐█ █▌      ▐▌  ▐▌
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌     ▐███▌ ▟██▖▐███ ▐▙██▖
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘       ▐▌█▐▌ ▘▄▟▌ ▐▌  ▐▛ ▐▌
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █        ▐▌▀▐▌▗█▀▜▌ ▐▌  ▐▌ ▐▌
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █        ▐▌ ▐▌▐▙▄█▌ ▐▙▄ ▐▌ ▐▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀        ▝▘ ▝▘ ▀▀▝▘  ▀▀ ▝▘ ▝▘
// ----------------------------------------------------------------

static PyObject *PyTensorBase_add_tensor_scalar(PyTensorBase *t, scalar s)
{
    PyTensorBase *result = PyTensorBase_create(t->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_add_tensor_scalar(&t->tb, s, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_add_tensor_tensor(PyTensorBase *a, PyTensorBase *b)
{
    TensorBase a_temp;
    TensorBase b_temp;

    if (TensorBase_broadcast_for_binop(&a->tb, &b->tb, &a_temp, &b_temp) < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Incompatible shapes for addition.");
        return NULL;
    }

    PyTensorBase *result = PyTensorBase_create(a_temp.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_add_tensor_tensor(&a_temp, &b_temp, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_add(PyObject *a, PyObject *b)
{
    // Valid types: PyTensorBase (with broadcastable dimensions), integers, floats
    // TODO: just get types and compare?

    if (!(can_math(a) && can_math(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }

    // PyTensorBase + (Long | Float)
    if (PyTensorBase_Check(a) && PyFloatOrLong_Check(b))
    {
        return PyTensorBase_add_tensor_scalar((PyTensorBase *)a, PyFloatOrLong_asDouble(b));
    }
    // (Long | Float) + PyTensorBase
    else if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_add_tensor_scalar((PyTensorBase *)b, PyFloatOrLong_asDouble(a));
    }
    // PyTensorBase + PyTensorBase
    else if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_add_tensor_tensor((PyTensorBase *)a, (PyTensorBase *)b);
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }
}

static PyObject *PyTensorBase_div_scalar_tensor(scalar s, PyTensorBase *t)
{
    PyTensorBase *result = PyTensorBase_create(t->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_div_scalar_tensor(s, &t->tb, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_divide(PyObject *a, PyObject *b)
{
    if (!(can_math(a) && can_math(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for division.");
        return NULL;
    }

    // (Long | Float) + PyTensorBase
    if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_div_scalar_tensor(PyFloatOrLong_asDouble(a), (PyTensorBase *)b);
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for division.");
        return NULL;
    }
}

static PyObject *PyTensorBase_negate(PyObject *a)
{
    PyTensorBase *result = PyTensorBase_create(((PyTensorBase *)a)->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_neg(&((PyTensorBase *)a)->tb, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_matrix_multiply(PyTensorBase *a, PyTensorBase *b)
{
    if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        ShapeArray new_shape = {0};

        if (TensorBase_get_matrix_multiplication_shape(&a->tb, &b->tb, &new_shape) < 0)
        {
            // printf("a->tb.shape: %ld, %ld\n", a->tb.shape[0], a->tb.shape[1]);
            // printf("b->tb.shape: %ld, %ld\n", b->tb.shape[0], b->tb.shape[1]);
            PyErr_SetString(PyExc_ValueError, "Incompatible shapes for matrix multiplication.");
            return NULL;
        }

        PyTensorBase *result = PyTensorBase_create(new_shape);
        if (!result)
        {
            // NOTE: error string set in PyTensorBase_create
            return NULL;
        }

        TensorBase_matrix_multiply(&a->tb, &b->tb, &result->tb);
        return (PyObject *)result;
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }
}

// ----------------------------------------------------------------
// ▗▄▄▄▖                      █
// ▐▛▀▀▘                ▐▌    ▀
// ▐▌   ▐▌ ▐▌▐▙██▖ ▟██▖▐███  ██   ▟█▙ ▐▙██▖▗▟██▖
// ▐███ ▐▌ ▐▌▐▛ ▐▌▐▛  ▘ ▐▌    █  ▐▛ ▜▌▐▛ ▐▌▐▙▄▖▘
// ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌    ▐▌    █  ▐▌ ▐▌▐▌ ▐▌ ▀▀█▖
// ▐▌   ▐▙▄█▌▐▌ ▐▌▝█▄▄▌ ▐▙▄ ▗▄█▄▖▝█▄█▘▐▌ ▐▌▐▄▄▟▌
// ▝▘    ▀▀▝▘▝▘ ▝▘ ▝▀▀   ▀▀ ▝▀▀▀▘ ▝▀▘ ▝▘ ▝▘ ▀▀▀
// ----------------------------------------------------------------

static PyObject *PyTensorBase_ones(PyModuleDef *module, PyObject *args)
{
    ShapeArray tb_shape = {0};
    if (args_to_shape(args, &tb_shape) < 0)
    {
        // NOTE: error message set in args_to_shape
        return NULL;
    }

    PyTensorBase *new_tb = PyTensorBase_create(tb_shape);
    // TODO: increment pointer?
    if (new_tb == NULL)
    {
        // NOTE: error message set in PyTensorBase_create
        return NULL;
    }

    for (long i = 0; i < new_tb->tb.numel; i++)
    {
        new_tb->tb.data[i] = 1;
    }

    return new_tb;
}

static PyObject *PyTensorBase_randn(PyModuleDef *module, PyObject *args)
{
    ShapeArray tb_shape = {0};
    if (args_to_shape(args, &tb_shape) < 0)
    {
        // NOTE: error message set in args_to_shape
        return NULL;
    }

    PyTensorBase *new_tb = PyTensorBase_create(tb_shape);
    if (new_tb == NULL)
    {
        // NOTE: error message set in PyTensorBase_create
        return NULL;
    }

    TensorBase_randn(&new_tb->tb, 0, 1);

    return new_tb;
}

static PyObject *PyTensorBase_sigmoid(PyModuleDef *module, PyObject *args)
{
    PyObject *obj;

    if (PyArg_ParseTuple(args, "O", &obj) < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Failed to parse TensorBase argument.");
        return NULL;
    }

    PyTensorBase *t = (PyTensorBase *)obj;
    // printf("t->tb.shape: %ld, %ld\n", t->tb.shape[0], t->tb.shape[1]);
    // printf("t->tb.numel: %ld\n", t->tb.numel);
    // printf("t->strides: %ld, %ld\n", t->tb.strides[0], t->tb.strides[1]);
    // printf("t->ndim: %ld\n", t->tb.ndim);
    PyTensorBase *result = PyTensorBase_create(t->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_sigmoid(&t->tb, &result->tb);
    return (PyObject *)result;
}

static PyMethodDef PyTensorBase_functions[] = {
    {"ones", (PyCFunction)PyTensorBase_ones, METH_VARARGS, "TODO: docs"},
    {"randn", (PyCFunction)PyTensorBase_randn, METH_VARARGS, "TODO: docs"},
    {"sigmoid", (PyCFunction)PyTensorBase_sigmoid, METH_VARARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
// ▗▄ ▄▖        ▗▖     ▗▄▖
// ▐█ █▌        ▐▌     ▝▜▌
// ▐███▌ ▟█▙  ▟█▟▌▐▌ ▐▌ ▐▌   ▟█▙
// ▐▌█▐▌▐▛ ▜▌▐▛ ▜▌▐▌ ▐▌ ▐▌  ▐▙▄▟▌
// ▐▌▀▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌ ▐▌  ▐▛▀▀▘
// ▐▌ ▐▌▝█▄█▘▝█▄█▌▐▙▄█▌ ▐▙▄ ▝█▄▄▌
// ▝▘ ▝▘ ▝▀▘  ▝▀▝▘ ▀▀▝▘  ▀▀  ▝▀▀
// ----------------------------------------------------------------

static PyModuleDef TensorBaseModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "match.tensorbase",
    .m_doc = PyDoc_STR("TODO: docs"),
    .m_size = -1,
    .m_methods = PyTensorBase_functions,
};

PyMODINIT_FUNC
PyInit_tensorbase(void)
{

    if (PyType_Ready(&PyTensorBaseType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&TensorBaseModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyTensorBaseType);
    if (PyModule_AddObject(m, "TensorBase", (PyObject *)&PyTensorBaseType) < 0)
    {
        Py_DECREF(&PyTensorBaseType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
