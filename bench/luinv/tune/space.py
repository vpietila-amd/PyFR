"""The lu_inv ``hip_opt`` knob space as a framework ``SearchSpace``.

Only the knobs that are safe keepers are exposed.  ``minblk``, ``pivot`` and
``ix32`` are the experiment knobs of the ``hip_opt`` variant; ``fence`` is left
out because it is an upper-bound probe that is racy as a blanket transform (see
the variant docstring).

A decoded framework config maps to the ``opts`` dict that
``hip_opt.variant_with_opts`` expects:

    {"minblk": 2}                      -> baseline launch bounds (bit-identical)
    {"minblk": 1, "pivot": "shfl"}     -> min-blocks 1 plus the shuffle arg-max
    {"ix32": "on"}                     -> 32-bit workspace offsets

The baseline config (min-blocks 2, no pivot/ix32 transform) reproduces the
frozen baseline kernel, so it is the genome the search confirms its finalists
against.
"""

# min-blocks the baseline kernel is built with (__launch_bounds__(256, 2))
BASELINE_MINBLK = 2

BASELINE_CONFIG = {'minblk': BASELINE_MINBLK, 'pivot': False, 'ix32': False}


def make_space(tuning_search, name='lu_inv_hip_opt'):
    """Build the ``SearchSpace`` over the hip_opt knobs."""
    Param = tuning_search.Param
    SearchSpace = tuning_search.SearchSpace

    params = (
        # __launch_bounds__ minimum blocks per CU; baseline is 2
        Param.ordinal('minblk', (1, 2, 3)),
        # replace the pivot arg-max LDS butterfly with a shuffle reduction
        Param.boolean('pivot'),
        # hoist the int64 workspace base so per-matrix offsets stay 32-bit
        Param.boolean('ix32'),
    )

    # minblk=1 and ix32 are performance substitutes, not an invalid pairing,
    # so no is_valid constraint is needed.
    return SearchSpace(params, name=name)


def decode_opts(config):
    """A framework knob config -> the ``hip_opt`` opts dict."""
    opts = {'minblk': int(config['minblk'])}
    if config.get('pivot'):
        opts['pivot'] = 'shfl'
    if config.get('ix32'):
        opts['ix32'] = 'on'
    return opts
