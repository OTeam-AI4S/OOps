# OOps
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OOps.

# OOps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OOps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://oops.valtech.ai/about/license-faq/>.

from __future__ import annotations

from . import generic

_SUPPORTED_DTYPES = generic._SUPPORTED_DTYPES
_cached_segment_execution_plan = generic._cached_segment_execution_plan
finish_long_reduction = generic.finish_long_reduction
flat_features = generic.flat_features
fused_reduction_csr = generic.fused_reduction_csr
prepare_long_reduction = generic.prepare_long_reduction
reduction_part_count = generic.reduction_part_count
resolve_execution_plan = generic.resolve_execution_plan
rows_per_edge = generic.rows_per_edge
schedule_supports = generic.schedule_supports
sum_csr = generic.sum_csr
supports_reduction_layout = generic.supports_reduction_layout
