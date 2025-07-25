from atscui.models.base_agent import BaseAgent

class SACAgent(BaseAgent):
    def _create_model(self):
        from stable_baselines3 import SAC
        return SAC(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            tensorboard_log=self.config.tensorboard_logs,
            verbose=self.config.verbose,
            ent_coef=self.config.ent_coef,
            gradient_steps=self.config.gradient_steps,
            target_entropy=self.config.target_entropy,
        )

    def train(self):
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )