# Upstream PR preparation

## slime PR 1: FSDP/SGLang weight-version publication

- Increment a trainer-side version only after each complete distributed update.
- Pass that version through `update_weights_from_distributed`.
- Add a RolloutManager method that returns all engine versions.
- Use SGLang's non-deprecated `/weight_version` endpoint.
- Tests: initial publication, monotonic updates, mixed-engine rejection and
  interrupted transfer behavior.

## slime PR 2: v0.2.1 compatibility fixes

- Keep Ring FlashAttention optional when context parallelism is one.
- Return an integer engine count from the no-new-engine branch.
- Tests: CP1 import without ring-flash-attn and rollout-engine reinitialization.

## Project-only integration

The Adaptive group scheduler, direct custom generation hook, update coordinator,
bounded driver and experiment tooling remain in this repository. They should not
be bundled into a compatibility PR before maintainers agree on public extension
points.

The exact reviewable diffs are in `patches/`. Both are applied to the locked
slime commit and pass reverse-apply checks. Before submission, rebase onto the
current upstream branch and replace project integration tests with the upstream
test harness.
