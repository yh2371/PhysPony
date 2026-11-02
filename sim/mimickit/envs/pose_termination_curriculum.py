# Schedule for the pose-termination radius and the power (torque) penalty during
# physics refinement. After a warmup at the env yaml's own values, the radius is
# annealed geometrically and the power coefficient linearly, keyed on training
# samples (the `Samples` column of the training log). Set `steps` for a
# piecewise-constant schedule instead.


SAMPLES_PER_HOUR = 437 * 131072  # ~57.3M samples/hour on an L40 at 4096 envs


class PoseTerminationCurriculum:
    """Anneals pose_termination_dist and power_coefficient during training."""

    def __init__(self, config, init_dist, init_power, num_envs):
        self._num_envs = int(num_envs)
        self._samples = float(config.get("start_samples", 0.0))

        # Warmup values: the env yaml's own settings.
        self._warmup_dist = float(init_dist)
        self._warmup_power = float(init_power)
        self._warmup_samples = float(config.get("warmup_samples", SAMPLES_PER_HOUR))

        self._steps = config.get("steps", None)
        if (self._steps is not None):
            self._steps = [(float(d), float(p)) for d, p in self._steps]
            default_span = len(self._steps) * SAMPLES_PER_HOUR
        else:
            self._start_dist = float(config.get("start_dist", 3.0))
            self._end_dist = float(config.get("end_dist", 0.5))
            self._start_power = float(config.get("start_power_coefficient", init_power))
            self._end_power = float(config.get("end_power_coefficient", 0.0003))
            default_span = 6 * SAMPLES_PER_HOUR

        self._anneal_samples = float(config.get("anneal_samples", default_span))
        assert(self._anneal_samples > 0.0), "anneal_samples must be positive"

        self._dist = self._warmup_dist
        self._power = self._warmup_power
        self._apply_progress()
        return

    def get_dist(self):
        return self._dist

    def get_power_coefficient(self):
        return self._power

    def get_samples(self):
        return self._samples

    def advance(self, num_steps=1):
        """Counts one control step per env, matching the logged Samples."""
        self._samples += num_steps * self._num_envs
        self._apply_progress()
        return

    def _apply_progress(self):
        t = (self._samples - self._warmup_samples) / self._anneal_samples
        if (t < 0.0):
            self._dist = self._warmup_dist
            self._power = self._warmup_power
            return

        t = min(t, 1.0)
        if (self._steps is not None):
            # Piecewise constant.
            # t == 1.0 must land on the last step, not past it.
            idx = min(int(t * len(self._steps)), len(self._steps) - 1)
            self._dist, self._power = self._steps[idx]
        else:
            # Geometric in distance, linear in power.
            self._dist = self._start_dist * (self._end_dist / self._start_dist) ** t
            self._power = self._start_power + t * (self._end_power - self._start_power)
        return


def build(config, init_dist, init_power, num_envs):
    if (config is None):
        return None
    return PoseTerminationCurriculum(config, init_dist, init_power, num_envs)
