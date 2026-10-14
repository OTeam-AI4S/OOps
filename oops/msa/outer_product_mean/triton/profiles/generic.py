import torch

from oops.core import CoreLaunchContext, CoreLaunchRule, CoreTileConfig, LaunchProfile

DUAL_PROJECTION_FALLBACK = LaunchProfile[CoreLaunchContext, CoreTileConfig](
    CoreTileConfig(
        name="outer_product_mean_dual_projection_generic",
        block_rows=16,
        block_reduction=64,
        num_warps=1,
        num_stages=1,
    ),
    rules=(
        CoreLaunchRule(
            CoreTileConfig(
                name="outer_product_mean_dual_projection_fp32_generic",
                block_rows=16,
                block_reduction=64,
                num_warps=2,
                num_stages=1,
            ),
            dtypes=(torch.float32,),
        ),
    ),
)


TAIL_FALLBACK = LaunchProfile[CoreLaunchContext, CoreTileConfig](
    CoreTileConfig(
        name="outer_product_mean_tail_generic",
        block_rows=64,
        block_channels=64,
        block_reduction=16,
        num_warps=4,
        num_stages=1,
    )
)
