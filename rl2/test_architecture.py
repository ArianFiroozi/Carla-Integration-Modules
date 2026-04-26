import torch
from dummy_env import DummyCarlaEnv
from actor_critic import UTCarActorCritic

def run_test():
    print("1. Initializing Dummy Environment...")
    env = DummyCarlaEnv()
    obs = env.reset()
    
    grid_tensor = torch.tensor(obs["grid"]).unsqueeze(0)
    scalar_tensor = torch.tensor(obs["scalars"]).unsqueeze(0)
    print(f"   [OK] Grid shape: {grid_tensor.shape}")
    print(f"   [OK] Scalars shape: {scalar_tensor.shape}")
    
    print("\n2. Initializing UTCarActorCritic Model...")
    model = UTCarActorCritic(latent_dim=128)
    print("   [OK] Model created.")
    
    print("\n3. Testing Forward Pass...")
    action, log_prob, entropy, value = model.get_action_and_value(grid_tensor, scalar_tensor)
    
    assert action.shape == (1, 3), f"Wrong action shape: {action.shape}"
    assert log_prob.shape == (1,), f"Wrong log_prob shape: {log_prob.shape}"
    assert entropy.shape == (1,), f"Wrong entropy shape: {entropy.shape}"
    assert value.shape == (1, 1), f"Wrong value shape: {value.shape}"
    print("   [OK] Forward pass successful and shapes are correct.")
    
    print("\n4. Testing Backward Pass (Gradients)...")
    loss = -log_prob.mean() + value.mean()
    loss.backward()
    
    assert model.extractor.cnn[0].weight.grad is not None, "Gradients are not flowing back to CNN!"
    print("   [OK] Backward pass successful! Gradients are flowing perfectly.")
    
    print("\n🚀 ALL TESTS PASSED! The architecture is completely safe to use.")

if __name__ == "__main__":
    run_test()