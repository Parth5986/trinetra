from config import EnvConfig
from env import EWenv


def main():

    print("1. Starting main()")

    config = EnvConfig(
        n_bands=10,
        n_emitters=3,
        max_steps=100,
        balanced_sampling=True,
    )

    print("2. Config created")

    env = EWenv(
        config=config,
        render_mode="human",
    )

    print("3. Environment created")

    observation, info = env.reset(seed=42)

    print("4. Environment reset completed")

    print("\n===== ENVIRONMENT STARTED =====")

    print("\nObservation shape:")
    print(observation.shape)

    print("\nSelected emitters:")
    print(info["emitter_modes"])

    total_reward = 0.0

    for step in range(20):

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        print(f"\nStep: {step + 1}")
        print(f"Action: Band {action}")
        print(f"Occupied: {info['band_occupied']}")
        print(f"Reward: {reward:.3f}")

        if terminated or truncated:
            break

    print("\n--- BAND ACTIVITY DEBUG ---")

    for emitter in env.emitters:
        frequencies = emitter.get_active_frequencies()
        power_dbm = emitter.get_signal_power_dbm(env.time_s)

        print(
            f"Mode={emitter.freq_mode} | "
            f"Freq={frequencies} MHz | "
            f"Power={power_dbm:.2f} dBm"
        )

    print("\n===== EPISODE FINISHED =====")
    print(f"Total Reward: {total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()