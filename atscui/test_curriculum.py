import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atscui.config.base_config import PPOConfig
from atscui.environment import createEnv
from atscui.models.agent_creator import createAlgorithm

def test_curriculum_learning():
    """
    A test script to verify the curriculum learning functionality.
    """
    print("--- Starting Curriculum Learning Test ---")

    # 1. Create a configuration object with curriculum learning enabled
    config = PPOConfig(
        # --- Basic file paths ---
        net_file="/Users/xnpeng/sumoptis/atscui/zfdx/net/zfdx.net.xml",
        # This will be used as a directory to save the generated file
        rou_file="/Users/xnpeng/sumoptis/atscui/zfdx/net/zfdx-perhour.rou.xml",
        csv_path="/Users/xnpeng/sumoptis/atscui/outs/test_curriculum_zfdx.csv",
        model_path="/Users/xnpeng/sumoptis/atscui/models/test_curriculum_zfdx_model.zip",
        eval_path="/Users/xnpeng/sumoptis/atscui/evals",
        predict_path="/Users/xnpeng/sumoptis/atscui/predicts",
        
        # --- Enable Curriculum Learning ---
        use_curriculum_learning=True,
        
        # --- Point to the new template file ---
        base_template_rou_file="/Users/xnpeng/sumoptis/atscui/zfdx/net/zfdx.rou.template.xml",
        
        # --- Curriculum Learning Parameters ---
        num_seconds=1000,             # Total simulation seconds for this test
        static_phase_ratio=0.5,       # Static: 500s, Dynamic: 500s
        base_flow_rate=200,           # Base flow for medium traffic
        dynamic_flows_rate=15,        # Rate for dynamic phase
        
        # --- Other training parameters ---
        total_timesteps=1200,         # Run for a short period to test
        algo_name="PPO",
        gui=False, # Disable GUI for faster testing
    )

    # 2. Create the environment
    # This should trigger the generation of 'curriculum.rou.xml'
    env = createEnv(config)

    # 3. Create the algorithm
    # The config object is converted to a dictionary. The createAlgorithm function
    # is now responsible for filtering the parameters.
    from dataclasses import asdict
    model_params = asdict(config)
    # We must remove algo_name as it is passed as a positional argument.
    model_params.pop('algo_name', None)

    model = createAlgorithm(env, config.algo_name, **model_params)

    # 4. Run the training for a few steps
    print("\n--- Starting model.learn() ---")
    model.learn(total_timesteps=config.total_timesteps)
    print("--- model.learn() completed successfully. ---")

    # 5. Close the environment
    env.close()
    print("--- Test script finished. ---")


if __name__ == "__main__":
    # Setup SUMO environment
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    
    test_curriculum_learning()
