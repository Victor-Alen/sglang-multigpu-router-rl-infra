from router.rl.simulator import OfflineGroupSimulator, SimulationScenario, bootstrap_mean_ci


def scenario(policy):
    return SimulationScenario(
        policy=policy,
        group_size=4,
        prompt_bucket=(128, 128),
        output_bucket=(64, 256),
        seed=7,
        groups=12,
        prefix_pool_size=2,
    )


def test_simulator_is_reproducible_and_reports_tail_metrics():
    simulator = OfflineGroupSimulator()
    first = simulator.run(scenario("adaptive-group")).summary()
    second = simulator.run(scenario("adaptive-group")).summary()
    for key in (
        "group_completion_seconds_mean",
        "group_completion_seconds_p95",
        "duplicated_prefill_tokens",
        "generated_tokens",
        "wall_seconds",
    ):
        assert first[key] == second[key]
    assert first["group_completion_seconds_p99"] >= first["group_completion_seconds_p95"]
    assert first["router_decision_ms_mean"] >= 0


def test_pack_and_split_expose_prefill_tradeoff():
    simulator = OfflineGroupSimulator()
    packed = simulator.run(scenario("fixed-pack")).summary()
    split = simulator.run(scenario("fixed-even-split")).summary()
    assert packed["duplicated_prefill_tokens"] <= split["duplicated_prefill_tokens"]


def test_bootstrap_interval_contains_constant_mean():
    assert bootstrap_mean_ci([3.0] * 10) == (3.0, 3.0)
