import jax

from MCTX.mctx  import MCTX, SearchConfig
from .scorch_nn import ScorchNeuralNetwork
from Go.env     import ScorchEnv

from typing import Tuple

class Scorch:
    def __init__(
        self,
        search_config : SearchConfig,
        env           : ScorchEnv,
        seed          : int = 42
    ) -> None:
        self.search_config : SearchConfig = search_config
        self.prng_key, init_key = jax.random.split(jax.random.PRNGKey(seed))

        self.neural_network : ScorchNeuralNetwork = ScorchNeuralNetwork()
        self.mctx           : MCTX                = MCTX(self.neural_network, env)

        self.params = self.neural_network.init(
            init_key,
            env.observation()[:1]
        )['params']

    def act(self, env : ScorchEnv, temperature : float=1.0) -> Tuple[jax.Array, jax.Array]:
        self.prng_key, search_key = jax.random.split(self.prng_key)

        return self.mctx.search(
            params      = self.params, 
            env         = env,
            config      = self.search_config,
            rng_key     = search_key,
            temperature = temperature
        )