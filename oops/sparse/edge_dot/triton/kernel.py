from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _bounded_node(node, num_nodes):

    return tl.maximum(tl.minimum(node.to(tl.int64), num_nodes - 1), 0)


@triton.jit
def _edge_dot_node_forward(
    edge_values_ptr,
    node_values_ptr,
    dst_index_ptr,
    out_ptr,
    outputs,
    num_nodes,
    groups: tl.constexpr,
    channels: tl.constexpr,
    use_float64: tl.constexpr,
    block_outputs: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    output_offsets = tl.program_id(0).to(tl.int64) * block_outputs + tl.arange(0, block_outputs)
    channel_offsets = tl.arange(0, block_channels)
    output_mask = output_offsets < outputs
    edge_offsets = output_offsets // groups
    group_offsets = output_offsets - edge_offsets * groups
    channel_mask = channel_offsets < channels

    dst = _bounded_node(tl.load(dst_index_ptr + edge_offsets, mask=output_mask, other=0), num_nodes)
    edge_base = edge_offsets[:, None] * groups * channels + group_offsets[:, None] * channels
    node_base = dst[:, None] * groups * channels + group_offsets[:, None] * channels
    offsets = channel_offsets[None, :]
    mask = output_mask[:, None] & channel_mask[None, :]

    edge_values = tl.load(edge_values_ptr + edge_base + offsets, mask=mask, other=0.0)
    node_values = tl.load(node_values_ptr + node_base + offsets, mask=mask & (num_nodes > 0), other=0.0)
    if use_float64:
        products = edge_values.to(tl.float64) * node_values.to(tl.float64)
        acc = tl.sum(products, axis=1)
    else:
        products = edge_values.to(tl.float32) * node_values.to(tl.float32)
        acc = tl.sum(products, axis=1)

    tl.store(out_ptr + output_offsets, acc, mask=output_mask)


@triton.jit
def _edge_dot_node_backward(
    grad_out_ptr,
    edge_values_ptr,
    node_values_ptr,
    dst_index_ptr,
    grad_edge_values_ptr,
    grad_node_values_ptr,
    outputs,
    num_nodes,
    groups: tl.constexpr,
    channels: tl.constexpr,
    need_grad_edge_values: tl.constexpr,
    need_grad_node_values: tl.constexpr,
    node_accumulate_fp32: tl.constexpr,
    block_outputs: tl.constexpr,
    block_channels: tl.constexpr,
    aggregate_local_nodes: tl.constexpr,
) -> None:

    output_offsets = tl.program_id(0).to(tl.int64) * block_outputs + tl.arange(0, block_outputs)
    channel_offsets = tl.arange(0, block_channels)
    output_mask = output_offsets < outputs
    edge_offsets = output_offsets // groups
    group_offsets = output_offsets - edge_offsets * groups
    channel_mask = channel_offsets < channels

    dst = _bounded_node(tl.load(dst_index_ptr + edge_offsets, mask=output_mask, other=0), num_nodes)
    edge_base = edge_offsets[:, None] * groups * channels + group_offsets[:, None] * channels
    node_base = dst[:, None] * groups * channels + group_offsets[:, None] * channels
    offsets = channel_offsets[None, :]
    mask = output_mask[:, None] & channel_mask[None, :]
    grad_out = tl.load(grad_out_ptr + output_offsets, mask=output_mask, other=0.0)

    if need_grad_edge_values:
        node_values = tl.load(
            node_values_ptr + node_base + offsets,
            mask=mask & (num_nodes > 0),
            other=0.0,
        )
        tl.store(
            grad_edge_values_ptr + edge_base + offsets,
            grad_out[:, None] * node_values,
            mask=mask,
        )

    if need_grad_node_values:
        edge_values = tl.load(edge_values_ptr + edge_base + offsets, mask=mask, other=0.0)
        contribution = grad_out[:, None] * edge_values
        if node_accumulate_fp32:
            contribution = grad_out.to(tl.float32)[:, None] * edge_values.to(tl.float32)
        if aggregate_local_nodes:
            first_node = tl.min(tl.where(output_mask, dst, num_nodes), axis=0)
            last_node = tl.max(tl.where(output_mask, dst, -1), axis=0)
            local_nodes = last_node - first_node + 1
            only_extremes = tl.sum((output_mask & (dst != first_node) & (dst != last_node)).to(tl.int32), axis=0) == 0
            two_destinations = (first_node != last_node) & only_extremes
            reductions = tl.where(two_destinations, 2, local_nodes)
            aggregate_nodes = (first_node == last_node) | (
                (outputs >= 4096) & (reductions <= 8) & (4 * reductions * groups <= block_outputs)
            )
            if aggregate_nodes:
                contribution = tl.where(mask, contribution, 0.0)
                for destination_offset in range(reductions):
                    destination = tl.where(
                        two_destinations,
                        tl.where(destination_offset == 0, first_node, last_node),
                        first_node + destination_offset,
                    )
                    selected = tl.where((dst == destination)[:, None], contribution, 0.0)
                    if block_outputs % groups == 0:
                        grouped = tl.reshape(selected, (block_outputs // groups, groups, block_channels))
                        aggregate = tl.sum(grouped, axis=0)
                        group_lanes = tl.arange(0, groups)
                        tl.atomic_add(
                            grad_node_values_ptr
                            + (destination * groups + group_lanes[:, None]) * channels
                            + channel_offsets[None, :],
                            aggregate,
                            mask=channel_mask[None, :] & (num_nodes > 0),
                            sem="relaxed",
                        )
                    else:
                        for group in tl.static_range(groups):
                            matches = output_mask & (group_offsets == group) & (dst == destination)
                            aggregate = tl.sum(tl.where(matches[:, None], contribution, 0.0), axis=0)
                            tl.atomic_add(
                                grad_node_values_ptr + (destination * groups + group) * channels + channel_offsets,
                                aggregate,
                                mask=channel_mask & (num_nodes > 0) & (tl.sum(matches.to(tl.int32), axis=0) > 0),
                                sem="relaxed",
                            )
            else:
                tl.atomic_add(
                    grad_node_values_ptr + node_base + offsets,
                    contribution,
                    mask=mask & (num_nodes > 0),
                    sem="relaxed",
                )
        else:
            tl.atomic_add(
                grad_node_values_ptr + node_base + offsets,
                contribution,
                mask=mask & (num_nodes > 0),
                sem="relaxed",
            )
