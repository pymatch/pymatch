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
// - info function for debugging

// [Reshaping PyTorch Tensors - DZone](https://dzone.com/articles/reshaping-pytorch-tensors)
// MATLAB Coder
// - https://www.mathworks.com/help/coder/ug/generate-code-with-implicit-expansion-enabled.html
// - https://www.mathworks.com/help/matlab/matlab_prog/compatible-array-sizes-for-basic-operations.html
// - https://www.mathworks.com/help/simulink/slref/coder.varsize.html

// - https://github.com/fulcrum-so/ziggy-pydust

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

// TODO: https://www.pcg-random.org/download.html

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

    scalar u1;
    do
    {
        u1 = (scalar)rand() / (scalar)((unsigned)RAND_MAX + 1);
    } while (u1 == 0);

    scalar u2 = (scalar)rand() / (scalar)((unsigned)RAND_MAX + 1);

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

    for (long i = 0; i < MAX_RANK; i++)
    {
        long dim = shape[i];
        // printf("loop: %d, dim: %ld, numel: %ld, ndim: %ld, shape: (%d, %d)\n", i, dim, td->numel, td->ndim, td->shape[0], td->shape[1]);
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

        // printf("loop: %d, dim: %ld, numel: %ld, ndim: %ld, shape: (%d, %d)\n", i, dim, td->numel, td->ndim, td->shape[0], td->shape[1]);
    }

    long stride = td->numel;
    for (long i = 0; i < td->ndim; i++)
    {
        stride /= td->shape[i];
        td->strides[i] = stride;
    }

    // NOTE: not initializing values
    td->data = malloc(td->numel * sizeof(scalar));

    // printf("init:: numel: %ld, data: %p, shape: (%d, %d)\n", td->numel, td->data, td->shape[0], td->shape[1]);

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

    // TODO: check for buffer overflow
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
    a_out->numel = a_in->numel;
    b_out->numel = b_in->numel;

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

    // printf("\n\na_ndim: %ld, b_ndim: %ld\n", a_out->ndim, b_out->ndim);
    // printf("a_out->shape: [%ld, %ld], b_out->shape: [%ld, %ld]\n", a_out->shape[0], a_out->shape[1], b_out->shape[0], b_out->shape[1]);
    // printf("a_out->numel: %ld, b_out->numel: %ld\n", a_out->numel, b_out->numel);
    // printf("a_out->strides: [%ld, %ld], b_out->strides: [%ld, %ld]\n", a_out->strides[0], a_out->strides[1], b_out->strides[0], b_out->strides[1]);

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

    // printf("a->ndim: %ld, b->ndim: %ld, out->ndim: %ld\n", a->ndim, b->ndim, out->ndim);
    // printf("a->shape: [%ld, %ld], b->shape: [%ld, %ld], out->shape: [%ld, %ld]\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1], out->shape[0], out->shape[1]);
    // printf("a->numel: %ld, b->numel: %ld, out->numel: %ld\n", a->numel, b->numel, out->numel);
    // printf("a->strides: [%ld, %ld], b->strides: [%ld, %ld], out->strides: [%ld, %ld]\n", a->strides[0], a->strides[1], b->strides[0], b->strides[1], out->strides[0], out->strides[1]);

    for (long i = 0; i < out->numel; i++)
    {
        // TODO: need a utility to make this faster (no need to two functions)
        index_to_indices(i, a->shape, a->ndim, a_indices);
        long a_index = indices_to_index(a_indices, a->strides, a->ndim);

        index_to_indices(i, b->shape, b->ndim, b_indices);
        long b_index = indices_to_index(b_indices, b->strides, b->ndim);

        // printf("a_index: %ld, b_index: %ld\n", a_index, b_index);
        // printf("a_indices: [%ld, %ld], b_indices: [%ld, %ld]\n", a_indices[0], a_indices[1], b_indices[0], b_indices[1]);

        out->data[i] = a->data[a_index] + b->data[b_index];
    }
}

// TODO: turn into TensorBase_binop_tensor_scalar?
static void TensorBase_add_tensor_scalar(TensorBase *t, scalar s, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        // TODO: check how much slower this is if the user passes in the
        // operator, and we put a switch statement in the loop (check godbolt?)
        out->data[i] = t->data[i] + s;
    }
}

static void TensorBase_div_scalar_tensor(scalar s, TensorBase *t, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = s / t->data[i];
    }
}

static void TensorBase_neg(TensorBase *t, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = -t->data[i];
    }
}

static void TensorBase_sigmoid(TensorBase *t, TensorBase *out)
{
    for (long i = 0; i < out->numel; i++)
    {
        out->data[i] = 1.0 / (1.0 + exp(-t->data[i]));
    }
}

static long TensorBase_get_matrix_multiplication_shape(TensorBase *a, TensorBase *b, ShapeArray *out)
{
    // TODO: relax the requirement for 2D tensors
    if (a->ndim != 2 || b->ndim != 2)
    {
        return -1;
    }

    if (a->shape[1] != b->shape[0])
    {
        return -1;
    }

    (*out)[0] = a->shape[0];
    (*out)[1] = b->shape[1];

    return 1;
}

// void mat_mul_jki(int n, float *A, float *B, float *C)
// {
//     for (int j = 0; j < n; j++)
//         for (int k = 0; k < n; k++)
//             for (int i = 0; i < n; i++)
//                 C[i + j * n] += A[i + k * n] * B[k + j * n];
// }


static void TensorBase_matrix_multiply(TensorBase *a, TensorBase *b, TensorBase *out)
{
    for (long i = 0; i < a->shape[0]; i++)
    {
        for (long j = 0; j < b->shape[1]; j++)
        {
            scalar sum = 0;
            for (long k = 0; k < a->shape[1]; k++)
            {
                sum += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
            out->data[i * out->shape[1] + j] = sum;
        }
    }
}
