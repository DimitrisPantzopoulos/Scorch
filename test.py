from Go.env import ScorchEnv
from Scorch.scorch_nn import ScorchNeuralNetwork
from MCTX.mctx import MCTX, SearchConfig

import jax
import jax.numpy as jnp



if __name__ == "__main__":
    env = ScorchEnv(batch_size=2)
    model = ScorchNeuralNetwork()

    init_key, search_key = jax.random.split(
        jax.random.PRNGKey(123)
    )

    params = model.init(
        init_key,
        env.observation(),
    )["params"]

    mcts = MCTX(
        neural_network=model,
        env=env,
    )

    # Search should leave the actual games unchanged.
    before = env.state

    config = SearchConfig(
        num_simulations=32,
        max_depth=None,
    )

    actions, policy_targets = mcts.search(
        params=params,
        env=env,
        config=config,
        rng_key=search_key,
        temperature=1.0,
    )
    
    assert actions.shape == (2,)
    assert policy_targets.shape == (2, 82)

    # Selected actions must be legal.
    game_ids = jnp.arange(env.batch_size)
    assert bool(env.legal_action_mask[game_ids, actions].all())

    # Training targets must be valid probability distributions.
    assert bool(jnp.isfinite(policy_targets).all())
    assert bool((policy_targets >= 0).all())
    assert bool(jnp.allclose(policy_targets.sum(axis=-1), 1.0))
    assert bool(jnp.all(
        jnp.where(env.legal_action_mask, 0.0, policy_targets) == 0.0
    ))

    # Search must not modify any field of the live state.
    assert all(
        bool(jnp.array_equal(old, new))
        for old, new in zip(
            jax.tree_util.tree_leaves(before),
            jax.tree_util.tree_leaves(env.state),
        )
    )

    print("Actions:", actions)
    print("Policy targets shape:", policy_targets.shape)
    print("Search checks passed.")

    # Apply the chosen actions to the actual games.
    env.step(actions)
    print("Players after stepping:", env.current_player)