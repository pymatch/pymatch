#include <stdlib.h>

// TODO:
// - Consider
//      - itemsize, nbytes, dtype
// - Methods
//      - item
//      - reshape, view
//      - broadcast (with alloc?)
//      - unbroadcast/reduce
//      - sum, mean, max, min, etc.
//      - sigmoid, relu, etc.
//      - add, sub, mul, div, etc.
//      - matmul, dot, etc.
//      - in-place operators

// [Reshaping PyTorch Tensors - DZone](https://dzone.com/articles/reshaping-pytorch-tensors)

// ----------------------------------------------------------------
//   ▄▄             ▄▄   █
//  █▀▀▌           ▐▛▀   ▀
// ▐▛    ▟█▙ ▐▙██▖▐███  ██   ▟█▟▌
// ▐▌   ▐▛ ▜▌▐▛ ▐▌ ▐▌    █  ▐▛ ▜▌
// ▐▙   ▐▌ ▐▌▐▌ ▐▌ ▐▌    █  ▐▌ ▐▌
//  █▄▄▌▝█▄█▘▐▌ ▐▌ ▐▌  ▗▄█▄▖▝█▄█▌
//   ▀▀  ▝▀▘ ▝▘ ▝▘ ▝▘  ▝▀▀▀▘ ▞▀▐▌
//                           ▜█▛▘
// ----------------------------------------------------------------

// To make it easier to change the data type later
typedef double scalar;

// NOTE: Assume maximum rank of 8
#define MAX_RANK 8
typedef long IndexArray[MAX_RANK];
typedef long ShapeArray[MAX_RANK];
typedef long StrideArray[MAX_RANK];

// ----------------------------------------------------------------
// ▗▖ ▗▖       █  ▗▄▖    █         █
// ▐▌ ▐▌ ▐▌    ▀  ▝▜▌    ▀   ▐▌    ▀
// ▐▌ ▐▌▐███  ██   ▐▌   ██  ▐███  ██   ▟█▙ ▗▟██▖
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▝█▄█▘ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
//  ▝▀▘   ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
// ----------------------------------------------------------------

static inline long max_long(long a, long b)
{
    return a > b ? a : b;
}

// Convert a linear index to a multi-dimensional index
// NOTE: this function does not check for out-of-bounds indices
static void index_to_indices(long index, ShapeArray shape, long ndim, IndexArray out)
{
    for (long i = ndim - 1; i >= 0; i--)
    {
        out[i] = index % shape[i];
        index /= shape[i];
    }
}

// Convert a multi-dimensional index to a linear index
// NOTE: this function does not check for out-of-bounds indices
static long indices_to_index(IndexArray indices, StrideArray strides, long ndim)
{
    long index = 0;
    for (long i = 0; i < ndim; i++)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

typedef struct
{
    scalar a;
    scalar b;
} randn_pair;

// Box-Muller method for generating normally distributed random numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#C++
randn_pair randn(scalar mu, scalar sigma)
{
    scalar two_pi = 2.0 * M_PI;

    scalar u1, u2;
    do
    {
        u1 = rand() / RAND_MAX;
    } while (u1 == 0);

    scalar mag = sigma * sqrt(-2.0 * log(u1));

    randn_pair result = {0};
    result.a = mag * cos(two_pi * u2) + mu;
    result.b = mag * sin(two_pi * u2) + mu;

    return result;
}

// ----------------------------------------------------------------
// ▗▄▄▄▖                              ▗▄▄▖
// ▝▀█▀▘                              ▐▛▀▜▌
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌     ▐▌ ▐▌ ▟██▖▗▟██▖ ▟█▙
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘       ▐███  ▘▄▟▌▐▙▄▖▘▐▙▄▟▌
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █        ▐▌ ▐▌▗█▀▜▌ ▀▀█▖▐▛▀▀▘
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █        ▐▙▄▟▌▐▙▄█▌▐▄▄▟▌▝█▄▄▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀        ▝▀▀▀  ▀▀▝▘ ▀▀▀  ▝▀▀
// ----------------------------------------------------------------

//
// A plain-c tensor data structure.
//
// This data structure lacks many of the features of a full-fledged tensor library,
// for example, it does not support:
// - multiple data types
// - GPU acceleration
// - SIMD acceleration
// - automatic differentiation (this is left to the Python interface)
//

// https://numpy.org/doc/stable/reference/arrays.ndarray.html
// Data structure for a tensor or single tensor item
// For a single tensor item
// - numel = 1
// - ndim = 0
// - shape = []
// - strides = ()
// - data: pointer to item (TODO: just store in pointer?)
// For a vector (1D tensor)
// - numel = length of data
// - ndim = 1
// - shape = [length of data]
// - strides = (1,)
// - data: pointer to data
typedef struct
{
    long numel;          // Number of elements (length of data array)
    long ndim;           // Number of dimensions
    ShapeArray shape;    // Shape of tensor
    StrideArray strides; // Strides of tensor
    scalar *data;        // Tensor data
} TensorBase;
// TODO: read-only for views of data?

static int TensorBase_init(TensorBase *td, ShapeArray shape)
{
    td->numel = 1;
    td->ndim = 0;

    // NOTE: not necessary, but good for documentation
    memset(td->shape, 0, sizeof(ShapeArray));
    memset(td->strides, 0, sizeof(StrideArray));

    long stride = td->numel;
    for (long i = 0; i < MAX_RANK; i++)
    {
        long dim = shape[i];
        if (dim == 0)
        {
            break;
        }
        if (dim < 0)
        {
            // All dimensions must be positive (zero indicates no-dimension)
            return -1;
        }
        td->numel *= dim;
        td->ndim += 1;
        td->shape[i] = dim;

        stride /= dim;
        td->strides[i] = stride;
    }

    // NOTE: not initializing values
    td->data = malloc(td->numel * sizeof(scalar));

    return td->data == NULL ? -1 : 0;
}

static void TensorBase_dealloc(TensorBase *td)
{
    free(td->data);
    td->data = NULL;
    td->numel = 0;
    td->ndim = 0;
    memset(td->shape, 0, sizeof(ShapeArray));
    memset(td->strides, 0, sizeof(StrideArray));
}

void TensorBase_to_string(TensorBase *td, char *buffer, size_t buffer_size)
{
    int bytes_written = snprintf(buffer, buffer_size, "[");
    buffer_size -= bytes_written;
    buffer += bytes_written;

    for (size_t index = 0; index < td->numel && buffer_size > 0; index++)
    {
        const char *sep = index < td->numel - 1 ? ", " : "";
        bytes_written = snprintf(buffer, buffer_size, "%f%s", td->data[index], sep);
        buffer_size -= bytes_written;
        buffer += bytes_written;
    }

    snprintf(buffer, buffer_size, "]");
}

void TensorBase_randn(TensorBase *td, scalar mu, scalar sigma)
{
    for (long i = 0; i < td->numel; i += 2)
    {
        randn_pair pair = randn(mu, sigma);
        td->data[i] = pair.a;
        if (i + 1 < td->numel)
        {
            td->data[i + 1] = pair.b;
        }
    }
}

// // TODO: turn into stringify
// void print_tensor(TensorBase *td, long current_dim, IndexArray indices)
// {
//     printf("[");
//     if (current_dim == td->ndim - 1)
//     {
//         // long index = TensorBase_indices_to_index(td, indices);
//         long index = indices_to_index(indices, td->strides, td->ndim);
//         print_vector(&td->data[index], td->shape[current_dim]);
//     }
//     else
//     {
//         for (long dim = 0; dim < td->shape[current_dim]; dim++)
//         {
//             IndexArray new_indices = {0};
//             memcpy(new_indices, indices, sizeof(IndexArray));
//             new_indices[current_dim] = dim;
//             print_tensor(td, current_dim + 1, new_indices);
//         }
//     }
//     printf("] ");
// }

// // TODO: change to accept output pointer?
// static void TensorBase_stringify(TensorBase *td)
// {
//     IndexArray indices = {0};
//     print_tensor(td, 0, indices);
//     printf("\n");
// }

// ----------------------------------------------------------------
// ▗▄▄▖                   ▗▖                      █
// ▐▛▀▜▌                  ▐▌                ▐▌    ▀
// ▐▌ ▐▌ █▟█▌ ▟█▙  ▟██▖ ▟█▟▌ ▟██▖ ▟██▖▗▟██▖▐███  ██  ▐▙██▖ ▟█▟▌
// ▐███  █▘  ▐▛ ▜▌ ▘▄▟▌▐▛ ▜▌▐▛  ▘ ▘▄▟▌▐▙▄▖▘ ▐▌    █  ▐▛ ▐▌▐▛ ▜▌
// ▐▌ ▐▌ █   ▐▌ ▐▌▗█▀▜▌▐▌ ▐▌▐▌   ▗█▀▜▌ ▀▀█▖ ▐▌    █  ▐▌ ▐▌▐▌ ▐▌
// ▐▙▄▟▌ █   ▝█▄█▘▐▙▄█▌▝█▄█▌▝█▄▄▌▐▙▄█▌▐▄▄▟▌ ▐▙▄ ▗▄█▄▖▐▌ ▐▌▝█▄█▌
// ▝▀▀▀  ▀    ▝▀▘  ▀▀▝▘ ▝▀▝▘ ▝▀▀  ▀▀▝▘ ▀▀▀   ▀▀ ▝▀▀▀▘▝▘ ▝▘ ▞▀▐▌
//                                                         ▜█▛▘
// ----------------------------------------------------------------

// Fill-in the shape and strides of the (temporary) output tensors
static long TensorBase_broadcast_for_binop(TensorBase *a_in, TensorBase *b_in, TensorBase *a_out, TensorBase *b_out)
{
    // NOTE: setting to zero as a sentinel value
    a_out->numel = 0;
    b_out->numel = 0;

    // The new (temporary) objects share data with the input objects
    a_out->data = a_in->data;
    b_out->data = b_in->data;

    long a_index = a_in->ndim - 1;
    long b_index = b_in->ndim - 1;
    long out_index = MAX_RANK - 1;

    long max_dim = max_long(a_in->ndim, b_in->ndim);
    a_out->ndim = max_dim;
    b_out->ndim = max_dim;

    // Fill final slots of out shape with zeros
    for (; out_index > max_dim - 1; out_index--)
    {
        a_out->shape[out_index] = 0;
        b_out->shape[out_index] = 0;
    }

    // Broadcast remaining shape dimensions
    for (; out_index >= 0; out_index--, a_index--, b_index--)
    {
        if ((a_index >= 0 && a_in->shape[a_index] == 1) || (a_index < 0 && b_index >= 0))
        {
            a_out->shape[out_index] = b_in->shape[b_index];
            b_out->shape[out_index] = b_in->shape[b_index];
        }
        else if ((b_index >= 0 && b_in->shape[b_index] == 1) || (b_index < 0 && a_index >= 0))
        {
            a_out->shape[out_index] = a_in->shape[a_index];
            b_out->shape[out_index] = a_in->shape[a_index];
        }
        else if (a_in->shape[a_index] == b_in->shape[b_index])
        {
            a_out->shape[out_index] = a_in->shape[a_index];
            b_out->shape[out_index] = a_in->shape[a_index];
        }
        else
        {
            // Incompatible shapes
            return -1;
        }
    }

    // Set stride of (temporary) tensors (use 0 for dimensions of size 1)
    long a_stride = a_out->numel;
    long b_stride = b_out->numel;

    for (out_index = 0; out_index < max_dim; out_index++)
    {
        long a_dim = a_out->shape[out_index];
        a_stride /= a_dim;
        a_out->strides[out_index] = a_dim > 0 ? a_stride : 0;

        long b_dim = b_out->shape[out_index];
        b_stride /= b_dim;
        b_out->strides[out_index] = b_dim > 0 ? b_stride : 0;
    }

    // Fill in remaining strides with zero
    for (; out_index < MAX_RANK; out_index++)
    {
        a_out->strides[out_index] = 0;
        b_out->strides[out_index] = 0;
    }

    return 1;
}

// ----------------------------------------------------------------
// ▗▖     █         ▄  ▗▄▖
// ▐▌     ▀        ▐█▌ ▝▜▌
// ▐▌    ██  ▐▙██▖ ▐█▌  ▐▌   ▟█▟▌
// ▐▌     █  ▐▛ ▐▌ █ █  ▐▌  ▐▛ ▜▌
// ▐▌     █  ▐▌ ▐▌ ███  ▐▌  ▐▌ ▐▌
// ▐▙▄▄▖▗▄█▄▖▐▌ ▐▌▗█ █▖ ▐▙▄ ▝█▄█▌
// ▝▀▀▀▘▝▀▀▀▘▝▘ ▝▘▝▘ ▝▘  ▀▀  ▞▀▐▌
//                           ▜█▛▘
// ----------------------------------------------------------------

// TODO: turn into TensorBase_binop_tensor_tensor?
static void TensorBase_add_tensor_tensor(TensorBase *a, TensorBase *b, TensorBase *out)
{
    IndexArray a_indices = {0};
    IndexArray b_indices = {0};

    for (long i = 0; i < out->numel; i++)
    {
        // TODO: need a utility to make this faster (no need to two functions)
        index_to_indices(i, a->shape, a->ndim, a_indices);
        long a_index = indices_to_index(a_indices, a->strides, a->ndim);

        index_to_indices(i, b->shape, b->ndim, b_indices);
        long b_index = indices_to_index(b_indices, b->strides, b->ndim);

        out->data[i] = a->data[a_index] + b->data[b_index];
    }
}

// TODO: turn into TensorBase_binop_tensor_scalar?
static void TensorBase_add_tensor_scalar(TensorBase *t, scalar s, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = t->data[i] + s;
    }
}
