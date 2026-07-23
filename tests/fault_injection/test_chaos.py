from integrations.chaos import OutputPortReject, schedule_partition_removal


def test_partition_rule_is_scoped_and_cleanup_is_idempotent() -> None:
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))

    partition = OutputPortReject("127.0.0.1", 15000, "run-1", uid=123, runner=runner)
    partition.install()
    partition.install()
    partition.remove()
    partition.remove()
    assert len(commands) == 2
    assert commands[0][0][:5] == ["sudo", "-n", "iptables", "-I", "OUTPUT"]
    assert "--uid-owner" in commands[0][0]
    assert "123" in commands[0][0]
    assert commands[1][0][3:5] == ["-D", "OUTPUT"]


def test_partition_removal_timer() -> None:
    removed = []
    commands = []

    def runner(command, **kwargs):
        commands.append(command)

    partition = OutputPortReject("127.0.0.1", 15000, "run-2", runner=runner)
    partition.install()
    timer = schedule_partition_removal(partition, 0.01, lambda: removed.append(True))
    timer.join(timeout=1)
    assert removed == [True]
    assert len(commands) == 2
