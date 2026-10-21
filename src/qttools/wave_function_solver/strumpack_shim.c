// Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

// LLM generated, TODO clean up.

/*
 * strumpack_shim.c -- thin C layer around STRUMPACK's C interface for ctypes.
 *
 * Why a shim: STRUMPACK_SparseSolver is passed by value, its precision /
 * index width are enums, and the distributed solver needs a (P+1) row
 * distribution array. The shim hides all of that behind a small opaque-handle
 * API, so Python only deals with a void* and plain arrays.
 *
 * Supports real (double) and complex (double complex) systems with either
 * 32-bit (int) or 64-bit (int64_t) indices. Unlike SuperLU_DIST, the index
 * width is a *runtime* choice (qt_create(use64=...)): STRUMPACK instantiates
 * both. Index arrays passed to qt_solve must match that width.
 *
 * Uses the MPI-distributed interface (STRUMPACK_MPI_DIST): rank r owns the
 * contiguous row block [fst_row, fst_row + m_loc), blocks ordered by rank.
 * The (P+1)-entry distribution array is assembled in the shim with an
 * MPI_Allgather, so Python only passes its own block.
 *
 * Debug output (read once, in qt_create):
 *   QT_STRUMPACK_VERBOSE=1  STRUMPACK's own output + factor stats
 *   QT_STRUMPACK_VERBOSE=2  additionally: per-phase timings (max over ranks)
 *                           for every solve call
 *
 * Options are injected from Python through qt_set_option() (see below), and
 * arbitrary STRUMPACK command line flags (e.g. "--sp_enable_gpu") can be
 * passed as a string to qt_create(); they are parsed by
 * STRUMPACK_set_from_options() before the explicit options are applied.
 *
 * The shim keeps a private copy of the local CSR block so that it never
 * depends on whether STRUMPACK copies the arrays it is given.
 *
 * Written against the STRUMPACK 7.2 C header (StrumpackSparseSolver.h). The
 * enum tables below reference STRUMPACK_AMD, _MMD, _AND, _MLF and _SPECTRAL
 * which are missing in older releases: delete those lines if the build
 * complains.
 *
 * Build (STRUMPACK built with MPI):
 *   mpicc -O2 -fPIC -shared strumpack_shim.c -o libqt_strumpack_shim.so \
 *       -I$STRUMPACK_DIR/include -L$STRUMPACK_DIR/lib \
 *       -lstrumpack -Wl,-rpath,$STRUMPACK_DIR/lib
 * (lib64 instead of lib on some systems.)
 */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "StrumpackSparseSolver.h"

enum { QT_DOFACT = 0, QT_SAMEPATTERN = 1, QT_FACTORED = 2 };
enum {
    QT_ERR_PATTERN = -1001,
    QT_ERR_NOFACTOR = -1002,
    QT_ERR_ALLOC = -1003,
    QT_ERR_DIST = -1004,
    QT_ERR_OVERFLOW = -1005
};

enum { QT_PH_REORDER = 0, QT_PH_FACTOR, QT_PH_SOLVE, QT_PH_COUNT };

typedef struct {
    int is_complex;
    int idx64;
    size_t scalar_size;
    size_t int_size;

    STRUMPACK_SparseSolver S;
    int solver_live;

    MPI_Comm comm;
    int rank, size;

    int factored; /* a numeric factorisation exists */
    int have_A;

    /* Private copy of the local CSR block + row distribution. */
    long long n, m_loc, fst_row, nnz_loc;
    void *nzval;
    void *colind;
    void *rowptr;
    void *dist; /* size+1 entries of the index type */
    int32_t m_loc32;
    int64_t m_loc64;

    /* Command line for STRUMPACK_init (kept alive for the handle's life). */
    int argc;
    char **argv;
    char *argbuf;

    int symmetric_pattern;
    int use_matsolve;

    int verbose; /* 0 quiet, 1 STRUMPACK output + stats, 2 + per-call timing */
    int options_printed;
    double phase[QT_PH_COUNT];
} qt_handle;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
static void put_idx(const qt_handle *h, void *arr, size_t i, long long v)
{
    if (h->idx64) ((int64_t *)arr)[i] = (int64_t)v;
    else ((int32_t *)arr)[i] = (int32_t)v;
}

static void release_A(qt_handle *h)
{
    free(h->nzval);
    free(h->colind);
    free(h->rowptr);
    free(h->dist);
    h->nzval = h->colind = h->rowptr = h->dist = NULL;
    h->have_A = 0;
}

static int store_A(qt_handle *h, long long n, long long m_loc,
                   long long fst_row, long long nnz_loc, const void *nz,
                   const void *ci, const void *rp)
{
    release_A(h);
    if (!h->idx64 && (n >= 2147483647LL || nnz_loc >= 2147483647LL))
        return QT_ERR_OVERFLOW;
    h->n = n;
    h->m_loc = m_loc;
    h->fst_row = fst_row;
    h->nnz_loc = nnz_loc;
    h->m_loc32 = (int32_t)m_loc;
    h->m_loc64 = (int64_t)m_loc;

    size_t nnz = (size_t)nnz_loc;
    h->nzval = malloc(nnz ? nnz * h->scalar_size : 1);
    h->colind = malloc(nnz ? nnz * h->int_size : 1);
    h->rowptr = malloc(((size_t)m_loc + 1) * h->int_size);
    if (!h->nzval || !h->colind || !h->rowptr) return QT_ERR_ALLOC;

    memcpy(h->nzval, nz, nnz * h->scalar_size);
    memcpy(h->colind, ci, nnz * h->int_size);
    memcpy(h->rowptr, rp, ((size_t)m_loc + 1) * h->int_size);
    return 0;
}

/* Collective: gather every rank's (fst_row, m_loc), check that the blocks are
 * contiguous and ordered by rank, and build the (size+1) dist array. The
 * outcome is identical on all ranks. */
static int build_dist(qt_handle *h)
{
    long long mine[2] = {h->fst_row, h->m_loc};
    long long *all = (long long *)malloc(2 * (size_t)h->size * sizeof(long long));
    if (!all) return QT_ERR_ALLOC;
    MPI_Allgather(mine, 2, MPI_LONG_LONG, all, 2, MPI_LONG_LONG, h->comm);

    int ok = 1;
    long long next = 0;
    for (int r = 0; r < h->size; ++r) {
        if (all[2 * r] != next) ok = 0;
        next += all[2 * r + 1];
    }
    if (next != h->n) ok = 0;

    if (ok) {
        h->dist = malloc(((size_t)h->size + 1) * h->int_size);
        if (!h->dist) {
            free(all);
            return QT_ERR_ALLOC;
        }
        for (int r = 0; r < h->size; ++r) put_idx(h, h->dist, r, all[2 * r]);
        put_idx(h, h->dist, h->size, h->n);
    }
    free(all);
    return ok ? 0 : QT_ERR_DIST;
}

static const void *local_rows_ptr(const qt_handle *h)
{
    return h->idx64 ? (const void *)&h->m_loc64 : (const void *)&h->m_loc32;
}

/* ------------------------------------------------------------------ */
/* Lifetime                                                            */
/* ------------------------------------------------------------------ */
/*
 * Collective. `args` is a whitespace separated list of STRUMPACK command line
 * flags (may be NULL or empty).
 */
void *qt_create(int is_complex, int use64, int fcomm, const char *args)
{
    qt_handle *h = (qt_handle *)calloc(1, sizeof(qt_handle));
    if (!h) return NULL;
    h->is_complex = is_complex;
    h->idx64 = use64 ? 1 : 0;
    h->scalar_size = is_complex ? 2 * sizeof(double) : sizeof(double);
    h->int_size = h->idx64 ? sizeof(int64_t) : sizeof(int32_t);
    h->comm = MPI_Comm_f2c((MPI_Fint)fcomm);
    MPI_Comm_rank(h->comm, &h->rank);
    MPI_Comm_size(h->comm, &h->size);

    const char *v = getenv("QT_STRUMPACK_VERBOSE");
    h->verbose = v ? atoi(v) : 0;

    /* argv = {"qt_strumpack", tokens of args..., NULL} */
    size_t len = args ? strlen(args) : 0;
    h->argbuf = (char *)malloc(len + 1);
    h->argv = (char **)calloc(len / 2 + 3, sizeof(char *));
    if (!h->argbuf || !h->argv) {
        free(h->argbuf);
        free(h->argv);
        free(h);
        return NULL;
    }
    if (len) memcpy(h->argbuf, args, len);
    h->argbuf[len] = '\0';
    h->argv[h->argc++] = "qt_strumpack";
    for (char *tok = strtok(h->argbuf, " \t\n"); tok;
         tok = strtok(NULL, " \t\n"))
        h->argv[h->argc++] = tok;

    STRUMPACK_PRECISION prec =
        is_complex ? (h->idx64 ? STRUMPACK_DOUBLECOMPLEX_64
                               : STRUMPACK_DOUBLECOMPLEX)
                   : (h->idx64 ? STRUMPACK_DOUBLE_64 : STRUMPACK_DOUBLE);

    STRUMPACK_init(&h->S, h->comm, prec, STRUMPACK_MPI_DIST, h->argc, h->argv,
                   h->verbose >= 1);
    STRUMPACK_set_from_options(h->S);
    h->solver_live = 1;
    return h;
}

/* Collective: must be called on every rank before MPI_Finalize. */
void qt_destroy(void *hp)
{
    qt_handle *h = (qt_handle *)hp;
    if (!h) return;
    if (h->solver_live) STRUMPACK_destroy(&h->S);
    release_A(h);
    free(h->argv);
    free(h->argbuf);
    free(h);
}

/* ------------------------------------------------------------------ */
/* Runtime option injection                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *name;
    int value;
} qt_enum_val;

static const qt_enum_val yesno_vals[] = {
    {"NO", 0}, {"YES", 1}, {"FALSE", 0}, {"TRUE", 1}, {NULL, 0}};
static const qt_enum_val compression_vals[] = {
    {"NONE", STRUMPACK_NONE},
    {"HSS", STRUMPACK_HSS},
    {"BLR", STRUMPACK_BLR},
    {"HODLR", STRUMPACK_HODLR},
    {"BLR_HODLR", STRUMPACK_BLR_HODLR},
    {"ZFP_BLR_HODLR", STRUMPACK_ZFP_BLR_HODLR},
    {"LOSSLESS", STRUMPACK_LOSSLESS},
    {"LOSSY", STRUMPACK_LOSSY},
    {NULL, 0}};
static const qt_enum_val reorder_vals[] = {
    {"NATURAL", STRUMPACK_NATURAL},
    {"METIS", STRUMPACK_METIS},
    {"PARMETIS", STRUMPACK_PARMETIS},
    {"SCOTCH", STRUMPACK_SCOTCH},
    {"PTSCOTCH", STRUMPACK_PTSCOTCH},
    {"RCM", STRUMPACK_RCM},
    {"GEOMETRIC", STRUMPACK_GEOMETRIC},
    {"AMD", STRUMPACK_AMD},
    {"MMD", STRUMPACK_MMD},
    {"AND", STRUMPACK_AND},
    {"MLF", STRUMPACK_MLF},
    {"SPECTRAL", STRUMPACK_SPECTRAL},
    {NULL, 0}};
static const qt_enum_val matching_vals[] = {
    {"NONE", STRUMPACK_MATCHING_NONE},
    {"MAX_CARDINALITY", STRUMPACK_MATCHING_MAX_CARDINALITY},
    {"MAX_SMALLEST_DIAGONAL", STRUMPACK_MATCHING_MAX_SMALLEST_DIAGONAL},
    {"MAX_SMALLEST_DIAGONAL_2", STRUMPACK_MATCHING_MAX_SMALLEST_DIAGONAL_2},
    {"MAX_DIAGONAL_SUM", STRUMPACK_MATCHING_MAX_DIAGONAL_SUM},
    {"MAX_DIAGONAL_PRODUCT_SCALING",
     STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING},
    {"COMBBLAS", STRUMPACK_MATCHING_COMBBLAS},
    {NULL, 0}};
static const qt_enum_val krylov_vals[] = {
    {"AUTO", STRUMPACK_AUTO},
    {"DIRECT", STRUMPACK_DIRECT},
    {"REFINE", STRUMPACK_REFINE},
    {"PREC_GMRES", STRUMPACK_PREC_GMRES},
    {"GMRES", STRUMPACK_GMRES},
    {"PREC_BICGSTAB", STRUMPACK_PREC_BICGSTAB},
    {"BICGSTAB", STRUMPACK_BICGSTAB},
    {NULL, 0}};
static const qt_enum_val gs_vals[] = {
    {"CLASSICAL", STRUMPACK_CLASSICAL}, {"MODIFIED", STRUMPACK_MODIFIED}, {NULL, 0}};

/* Parse "123" or a symbolic name (case-insensitive). 0 on success. */
static int parse_value(const qt_enum_val *tbl, const char *s, int *out)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (end != s && *end == '\0') {
        *out = (int)v;
        return 0;
    }
    if (tbl)
        for (; tbl->name; ++tbl)
            if (strcasecmp(tbl->name, s) == 0) {
                *out = tbl->value;
                return 0;
            }
    return -1;
}

static int parse_double(const char *s, double *out)
{
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end != '\0') return -1;
    *out = v;
    return 0;
}

/*
 * Set one STRUMPACK option by name (value as number string or enum name).
 * Call before the first solve. Returns 0 on success, -1 unknown option,
 * -2 invalid value, -3 option is managed by the shim (Verbose).
 *
 * Enum/choice options: Compression, ReorderingMethod, Matching, KrylovSolver,
 *   GramSchmidt
 * Boolean options:     UseGPU, MetisNodeNDP, SymmetricPattern, UseMatsolve
 * Integer options:     MaxIt, GMRESRestart, NDParam, CompressionMinSepSize,
 *   CompressionMinFrontSize, CompressionLeafSize, CompressionButterflyLevels,
 *   CompressionLossyPrecision
 * Floating options:    RelTol, AbsTol, CompressionRelTol, CompressionAbsTol
 */
int qt_set_option(void *hp, const char *name, const char *value)
{
    qt_handle *h = (qt_handle *)hp;
    int v;
    double d;

    if (!strcmp(name, "Verbose")) return -3;

#define QT_ENUM(NAME, TABLE, CALL)                                             \
    if (!strcmp(name, NAME)) {                                                 \
        if (parse_value(TABLE, value, &v)) return -2;                          \
        CALL;                                                                  \
        return 0;                                                              \
    }
#define QT_DBL(NAME, CALL)                                                     \
    if (!strcmp(name, NAME)) {                                                 \
        if (parse_double(value, &d)) return -2;                                \
        CALL;                                                                  \
        return 0;                                                              \
    }

    QT_ENUM("Compression", compression_vals,
            STRUMPACK_set_compression(h->S, (STRUMPACK_COMPRESSION_TYPE)v))
    QT_ENUM("ReorderingMethod", reorder_vals,
            STRUMPACK_set_reordering_method(h->S,
                                            (STRUMPACK_REORDERING_STRATEGY)v))
    QT_ENUM("Matching", matching_vals,
            STRUMPACK_set_matching(h->S, (STRUMPACK_MATCHING_JOB)v))
    QT_ENUM("KrylovSolver", krylov_vals,
            STRUMPACK_set_Krylov_solver(h->S, (STRUMPACK_KRYLOV_SOLVER)v))
    QT_ENUM("GramSchmidt", gs_vals,
            STRUMPACK_set_GramSchmidt_type(h->S,
                                           (STRUMPACK_GRAM_SCHMIDT_TYPE)v))

    QT_ENUM("UseGPU", yesno_vals,
            v ? STRUMPACK_enable_gpu(h->S) : STRUMPACK_disable_gpu(h->S))
    QT_ENUM("MetisNodeNDP", yesno_vals,
            v ? STRUMPACK_enable_METIS_NodeNDP(h->S)
              : STRUMPACK_disable_METIS_NodeNDP(h->S))
    QT_ENUM("SymmetricPattern", yesno_vals, h->symmetric_pattern = v)
    QT_ENUM("UseMatsolve", yesno_vals, h->use_matsolve = v)

    QT_ENUM("MaxIt", NULL, STRUMPACK_set_maxit(h->S, v))
    QT_ENUM("GMRESRestart", NULL, STRUMPACK_set_gmres_restart(h->S, v))
    QT_ENUM("NDParam", NULL, STRUMPACK_set_nd_param(h->S, v))
    QT_ENUM("CompressionMinSepSize", NULL,
            STRUMPACK_set_compression_min_sep_size(h->S, v))
    QT_ENUM("CompressionMinFrontSize", NULL,
            STRUMPACK_set_compression_min_front_size(h->S, v))
    QT_ENUM("CompressionLeafSize", NULL,
            STRUMPACK_set_compression_leaf_size(h->S, v))
    QT_ENUM("CompressionButterflyLevels", NULL,
            STRUMPACK_set_compression_butterfly_levels(h->S, v))
    QT_ENUM("CompressionLossyPrecision", NULL,
            STRUMPACK_set_compression_lossy_precision(h->S, v))

    QT_DBL("RelTol", STRUMPACK_set_rel_tol(h->S, d))
    QT_DBL("AbsTol", STRUMPACK_set_abs_tol(h->S, d))
    QT_DBL("CompressionRelTol", STRUMPACK_set_compression_rel_tol(h->S, d))
    QT_DBL("CompressionAbsTol", STRUMPACK_set_compression_abs_tol(h->S, d))
#undef QT_ENUM
#undef QT_DBL
    return -1;
}

/* ------------------------------------------------------------------ */
/* Solve                                                               */
/* ------------------------------------------------------------------ */
/*
 * Solve A x = b with A in distributed CSR (rows fst_row .. fst_row+m_loc-1,
 * global column indices, local row pointer starting at 0, index arrays of the
 * width chosen in qt_create) and b / x the local blocks (m_loc x nrhs, column
 * major, leading dimension ld >= m_loc). b is not modified.
 *
 * mode: QT_DOFACT        analyse + factor + solve
 *       QT_SAMEPATTERN   same sparsity pattern as the last factorisation: keep
 *                        the ordering/symbolic analysis, refactor, solve
 *       QT_FACTORED      only solve with the stored factorisation (A and the
 *                        matrix arguments are ignored)
 *
 * Returns 0 on success, a positive STRUMPACK_RETURN_CODE (1 matrix not set,
 * 2 reordering error, 3 zero pivot, 4 no convergence, 5 inaccurate inertia),
 * or a negative QT_ERR_* code. Negative codes and the pattern check are
 * decided collectively, so every rank returns the same code.
 */
int qt_solve(void *hp, int mode, long long n, long long m_loc,
             long long fst_row, long long nnz_loc, const void *nzval,
             const void *colind, const void *rowptr, const void *b, void *x,
             int ld, int nrhs)
{
    qt_handle *h = (qt_handle *)hp;
    int rc;
    STRUMPACK_RETURN_CODE sc;

    static const char *mode_names[] = {"DOFACT", "SamePattern", "FACTORED"};

    memset(h->phase, 0, sizeof(h->phase));

    if (mode == QT_FACTORED) {
        if (!h->factored || !h->have_A) return QT_ERR_NOFACTOR;
    } else {
        if (mode == QT_SAMEPATTERN && h->factored) {
            int bad = h->n != n || h->m_loc != m_loc ||
                      h->fst_row != fst_row || h->nnz_loc != nnz_loc ||
                      memcmp(h->rowptr, rowptr,
                             ((size_t)m_loc + 1) * h->int_size) != 0 ||
                      memcmp(h->colind, colind,
                             (size_t)nnz_loc * h->int_size) != 0;
            int any_bad;
            MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, h->comm);
            if (any_bad) return QT_ERR_PATTERN;
        } else {
            mode = QT_DOFACT;
        }

        if (mode == QT_DOFACT && h->factored) {
            STRUMPACK_delete_factors(h->S);
            h->factored = 0;
        }

        rc = store_A(h, n, m_loc, fst_row, nnz_loc, nzval, colind, rowptr);
        if (rc) return rc;

        if (mode == QT_DOFACT) {
            rc = build_dist(h);
            if (rc) return rc;
            STRUMPACK_set_distributed_csr_matrix(
                h->S, local_rows_ptr(h), h->rowptr, h->colind, h->nzval,
                h->dist, h->symmetric_pattern);
        } else {
            /* Same pattern => same distribution; dist was kept from the
             * DOFACT call... but store_A released it, so rebuild. */
            rc = build_dist(h);
            if (rc) return rc;
            STRUMPACK_update_distributed_csr_matrix_values(
                h->S, local_rows_ptr(h), h->rowptr, h->colind, h->nzval,
                h->dist, h->symmetric_pattern);
        }
    }

    if (h->verbose >= 2 && h->rank == 0) {
        printf("[qt_strumpack] enter solve mode=%s n=%lld nrhs=%d\n",
               mode_names[mode], (long long)h->n, nrhs);
        fflush(stdout);
    }

    double t_start = MPI_Wtime();
    int info = 0;

    if (mode != QT_FACTORED) {
        double t0 = MPI_Wtime();
        if (mode == QT_DOFACT) {
            sc = STRUMPACK_reorder(h->S);
            if (sc != STRUMPACK_SUCCESS) {
                h->factored = 0;
                return (int)sc;
            }
        }
        h->phase[QT_PH_REORDER] = MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        sc = STRUMPACK_factor(h->S);
        h->phase[QT_PH_FACTOR] = MPI_Wtime() - t0;
        if (sc != STRUMPACK_SUCCESS) {
            /* State after a failed factorisation is unreliable: force a
             * fresh analysis next time. */
            h->factored = 0;
            return (int)sc;
        }
        h->factored = 1;
    }

    {
        double t0 = MPI_Wtime();
        if (h->use_matsolve && nrhs > 1) {
            sc = STRUMPACK_matsolve(h->S, nrhs, b, ld, x, ld, 0);
            if (sc != STRUMPACK_SUCCESS) info = (int)sc;
        } else {
            for (int j = 0; j < nrhs; ++j) {
                size_t off = (size_t)j * (size_t)ld * h->scalar_size;
                sc = STRUMPACK_solve(h->S, (const char *)b + off,
                                     (char *)x + off, 0);
                if (sc != STRUMPACK_SUCCESS && !info) info = (int)sc;
            }
        }
        h->phase[QT_PH_SOLVE] = MPI_Wtime() - t0;
    }
    double dt = MPI_Wtime() - t_start;

    /* These queries are collective in the MPI-distributed solver: every rank
     * must call them, only rank 0 prints. */
    int its = 0;
    if (h->verbose >= 1) {
        long long fnz = STRUMPACK_factor_nonzeros(h->S);
        long long fmem = STRUMPACK_factor_memory(h->S);
        its = STRUMPACK_its(h->S);
        if (mode != QT_FACTORED && h->rank == 0) {
            printf("[qt_strumpack] factor nonzeros=%lld memory=%lld bytes\n",
                   fnz, fmem);
            fflush(stdout);
        }
    }

    if (h->verbose >= 2) {
        double tmax_total, tmax[QT_PH_COUNT];
        MPI_Reduce(&dt, &tmax_total, 1, MPI_DOUBLE, MPI_MAX, 0, h->comm);
        MPI_Reduce(h->phase, tmax, QT_PH_COUNT, MPI_DOUBLE, MPI_MAX, 0,
                   h->comm);
        if (h->rank == 0) {
            printf("[qt_strumpack] mode=%s n=%lld nrhs=%d info=%d "
                   "total=%.4fs\n",
                   mode_names[mode], (long long)h->n, nrhs, info, tmax_total);
            printf("[qt_strumpack]   reorder=%.4fs factor=%.4fs "
                   "solve=%.4fs (its=%d)\n",
                   tmax[QT_PH_REORDER], tmax[QT_PH_FACTOR], tmax[QT_PH_SOLVE],
                   its);
            fflush(stdout);
        }
    }
    return info;
}