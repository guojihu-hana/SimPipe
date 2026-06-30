from simpipe.pipeline.scheduling.base import ScheduleContext, ScheduleStrategy
from simpipe.core.types import Schedule

# OctoPipe uses dynamic OrderedQueue scheduling at runtime; no static plan.


class OctoPipeStrategy(ScheduleStrategy):
    name = Schedule.OctoPipe

    def generate(self, ctx: ScheduleContext):
        return None  # dynamic
